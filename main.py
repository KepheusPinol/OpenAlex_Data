# %%
import difflib
import nltk
import ijson
import pyalex
from pyalex import Works, Sources, config
import json
import re
import math
import langcodes
from collections import Counter
from itertools import chain, islice
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, shared_memory
import pickle
import gzip

# Sicherstellen, dass die Stopwörter heruntergeladen sind
nltk.download('stopwords')

# Initialisierung mit einem festgelegten Seed für konsistente Ergebnisse in 'langdetect'
DetectorFactory.seed = 0

# Konstante Konfigurationswerte
EMAIL = "chr.brenscheidt@gmail.com"
RETRY_HTTP_CODES = [429, 500, 503]

def setup_pyalex():
    """ Setup pyalex configuration. """
    pyalex.config.email = EMAIL
    config.max_retries = 0
    config.retry_backoff_factor = 0.1
    config.retry_http_codes = RETRY_HTTP_CODES
def process_page(page, ids_set):
    """
    Verarbeitet eine einzelne Seite von Veröffentlichungen und sammelt die Ergebnisse in einem Batch.

    Args:
        page: Eine Seite mit Veröffentlichungen.
        ids_set: Ein Set mit bereits verarbeiteten IDs, um Duplikate zu vermeiden.

    Returns:
        Ein Batch (Liste) mit den verarbeiteten Veröffentlichungen.
    """
    batch = []  # Zwischenspeicher für verarbeitete Veröffentlichungen der aktuellen Seite
    for publication in page:
        # ID aufbereiten
        publication_id = publication['id'].replace("https://openalex.org/", "")
        if publication_id in ids_set:  # Duplikat schnell überspringen
            continue

        # Autoren extrahieren
        referenced_works_id = [
            ref_id.replace("https://openalex.org/", "")
            for ref_id in publication.get('referenced_works', [])
        ]

        # Publikation zusammenstellen
        batch.append({
            'id': publication_id,
            'title': publication['title'],
            'abstract': publication.get("abstract", ""),
            'language': publication.get('language', ""),
            'kombinierte Terme Titel und Abstract': {},
            'referenced_works_count': publication.get('referenced_works_count', 0),
            'referenced_works': referenced_works_id,
            'cited_by_count': publication.get('cited_by_count', 0),
            'referencing_works': [],
            'kombinierte Terme referencing_works': {},
            'count_reference': "",
            'reference_works': [],
            'kombinierte Terme referenced_works': {},
            'count_co_referenced': "",
            'co_referenced_works': [],
            'count_co_referencing': "",
            'co_referencing_works': [],
            'count_co_reference': "",
            'co_reference_works': []
        })

    # IDs zum eindeutigen Set hinzufügen (lokal im Thread)
    ids_set.update(pub['id'] for pub in batch)
    return batch


def extract_publication_data_parallel(pager, base_publications_unique):
    """
    Parallelisierte Verarbeitung von Veröffentlichungsdaten.

    Args:
        pager: Pagination-Objekt, das Seiten mit Veröffentlichungen liefert.
        base_publications_unique: Existierende Liste von Veröffentlichungen (bereits verarbeitete Publikationen enthalten).

    Returns:
        Aktualisierte Liste `base_publications_unique` mit allen verarbeiteten Veröffentlichungen.
    """
    # Bestehende IDs in ein Set laden
    ids_set = set(item['id'] for item in base_publications_unique)

    # ThreadPool für die parallele Verarbeitung der Seiten
    with ThreadPoolExecutor() as executor:
        futures = []
        for page in pager.paginate(per_page=200, n_max=None):
            # Übergibt jede Seite an einen eigenen Thread
            futures.append(executor.submit(process_page, page, ids_set))

        # Ergebnisse aus allen Threads sammeln
        for future in futures:
            batch = future.result()
            base_publications_unique.extend(batch)  # Ergebnisse zur Hauptliste hinzufügen

    return base_publications_unique


def get_by_api(pager, filename):
    # all_publications_unique enthält einmalig die Metadaten aller Ausgangspublikationen
    base_publications_unique = extract_publication_data_parallel(pager, [])

    base_publications_unique.sort(key=lambda x: x.get('id', 0), reverse=True)
    save_to_json(filename, base_publications_unique)

    print(f"Anzahl der Ausgangspublikationen: {len(base_publications_unique)}")

    return base_publications_unique


def get_referenced_works(base_publications_unique):
    # referenced_publications_unique enthält einmalig die Metadaten aller zitierten Publikationen der Ausgangspublikationen
    unique_referenced_pubs = []
    referenced_publications_unique_ids = []
    referenced_works_count = 0

    for publication in base_publications_unique:
        referenced_publications_unique_ids = merge_and_deduplicate(referenced_publications_unique_ids, publication['referenced_works'])
    pager_referenced = build_pager(referenced_publications_unique_ids)

    for pager in pager_referenced:
        unique_referenced_pubs = extract_publication_data_parallel(pager, unique_referenced_pubs)
        #unique_referenced_pubs.sort(key=lambda x: x.get('id', 0), reverse=True)
        referenced_works_count += 50
        print('Es wurden ', referenced_works_count, ' von ', len(referenced_publications_unique_ids), ' zitierten Dokumente abgerufen.')

    print(f"Anzahl der unique referenced publications: {len(unique_referenced_pubs)}")

    return unique_referenced_pubs

def build_pager(list_ids):
    pagers = []

    # Durch die ID-Liste in Schritten von 50 iterieren
    for i in range(0, len(list_ids), 50):
        # Erstellen eines ref_id Strings aus den aktuellen 50 IDs (oder weniger, wenn nicht mehr verfügbar)
        ref_id = '|'.join(str(id) for id in list_ids[i:i + 50])

        # Erstellen eines pager-Objekts mit den aktuellen IDs
        pager = Works().filter(ids={"openalex": ref_id}).select([
            "id", "title", "referenced_works",
            "abstract_inverted_index", "cited_by_count", "referenced_works_count", "language"])

        # Hinzufügen des pager-Objekts zur pagers-Liste
        pagers.append(pager)

    return pagers

from concurrent.futures import ThreadPoolExecutor
from threading import Lock


def get_referencing_works(base_publications_unique):
    referencing_pub_unique = []
    count_pub = 0
    lock = Lock()  # Zum sicheren Zugriff auf count_pub

    def process_publication(publication):
        nonlocal count_pub
        pager_referencing = Works().filter(cites=publication['id']).select(
            ["id", "title", "referenced_works", "abstract_inverted_index", "cited_by_count", "referenced_works_count",
             "language"])
        referencing_publications = extract_publication_data_parallel(pager_referencing, [])
        referencing_publications_ids = [pub['id'] for pub in referencing_publications]
        publication['referencing_works'] = referencing_publications_ids
        publication['reference_works'] = merge_and_deduplicate(publication['referenced_works'],
                                                               publication['referencing_works'])
        publication['count_reference'] = len(publication['reference_works'])

        # Thread-sicherer Zugriff auf count_pub
        with lock:
            count_pub += 1
            print(f"Es wurden {count_pub} von {len(base_publications_unique)} abgerufen")

        return referencing_publications

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_publication, base_publications_unique))

    for referencing_publications in results:
        referencing_pub_unique += referencing_publications

    print(f"Anzahl der unique referencing publications: {len(referencing_pub_unique)}")
    return referencing_pub_unique



def process_publication(publication):
    if publication['language'] == 'en':
        term_dict = initialize_term_dict_metadata(
            texttransformation_metadata(
                f"{publication['title']} {publication['abstract']}"
            )
        )
        sorted_term_dict = dict(sorted(term_dict.items()))
        publication['kombinierte Terme Titel und Abstract'] = sorted_term_dict
    else:
        publication['kombinierte Terme Titel und Abstract'] = {}
    return publication

from concurrent.futures import as_completed

def indexing_metadata(base_publications_unique):
    with ProcessPoolExecutor() as executor:
        # Aufträge erstellen
        futures = {executor.submit(process_publication, pub): pub for pub in base_publications_unique}

        # Fortschrittsanzeige mit tqdm
        progress = tqdm(total=len(base_publications_unique), desc="Bearbeitungsstand", unit="Publikationen")

        processed = []

        # Fortschritt nachverfolgen
        for future in as_completed(futures):
            processed.append(future.result())
            progress.update(1)  # Fortschritt um 1 erhöhen

        progress.close()

    return processed

def get_stopwords_for_language(language):
    """
    Holt die Stopwords für die gegebene Sprache. Standardmäßig wird Englisch verwendet, falls die Sprache nicht unterstützt wird.
    """
    try:
        return set(stopwords.words(language))
    except OSError:
        # Standardmäßig Englisch verwenden, falls die Sprache nicht unterstützt wird
        return set(stopwords.words('english'))


def get_stemmer_for_language(language):
    """
    Gibt den Stemmer für die gegebene Sprache zurück. Standardmäßig wird Englisch verwendet, falls die Sprache nicht unterstützt wird.
    """
    try:
        return SnowballStemmer(language)
    except ValueError:
        # Standardmäßig Englisch verwenden, falls die Sprache nicht unterstützt wird
        return SnowballStemmer('english')


def texttransformation_metadata(text):
    if text is None:
        text = ""

    try:
        # Erkennung der Sprache des Textes
        language_code = detect(text)
        language_name = langcodes.get(language_code).language_name()
    except LangDetectException:
        language_name = 'English'  # Standardmäßig Englisch verwenden

    # Holen der Stopwords für die erkannte Sprache
    stop_words = get_stopwords_for_language(language_name)
    stemmer = get_stemmer_for_language(language_name)

    # Text in Kleinbuchstaben umwandeln
    text = text.lower()

    # Nicht-Wörter entfernen außer
    text = re.sub(r'[^\w\s\-@]', '', text)

    # Wörter splitten, Stopwörter entfernen und stemmen
    words = [
        word for word in text.split()
        if len(word) >= 3 and not re.match(r'^[\d\W]', word) and not re.fullmatch(r'[\d\W]+', word)
    ]
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]

    return filtered_words


def initialize_term_dict_metadata(terms):
    """
    :param terms: Eine Liste von Begriffen, die gezählt werden sollen.
    :return: Eine Liste von Dictionaries, wobei jedes Dictionary einen einzigartigen Begriff und dessen Zählwert enthält.
     """

    # Dictionary zur Speicherung der Zählwerte
    term_count_dict = {}

    # Begriffe zählen
    for term in terms:
        if term not in term_count_dict:
            term_count_dict[term] = 1

    return term_count_dict

def count_terms(publication):
    """
    Hilfsfunktion zur parallelen Zählung der Terme in einer einzelnen Publikation.
    Gibt einen `Counter` für diese Publikation zurück.
    """
    return Counter(publication["kombinierte Terme Titel und Abstract"].keys())


def document_frequency(publications_unique, num_documents):
    """
    Optimierte kombinierte Funktion:
    - Nutzung von `collections.Counter` für effiziente Zählung.
    - Parallelisierung mit `multiprocessing` Pool für große Datenmengen.
    """

    # Parallele Verarbeitung jeder Publikation
    with Pool() as pool:
        # Zähle Begriffe in parallelen Prozessen, Rückgabe ist eine Liste von Counters
        partial_counts = pool.map(count_terms, publications_unique)

    # Summiere alle erzeugten Counter zu einem einzigen
    document_frequency_dict = sum(partial_counts, Counter())

    # Entferne Terme mit Frequenz 1 oder größer als num_documents / 3
    filtered_df = {
        term: count
        for term, count in document_frequency_dict.items()
        if count != 1 and count <= num_documents / 3  # Parameter für die Filterung
    }

    # Sortiere die Ergebnisse nach Frequenz in absteigender Reihenfolge
    sorted_df = dict(sorted(filtered_df.items(), key=lambda item: item[1], reverse=True))

    # Speichere die Ergebnisse in einer JSON-Datei
    save_to_json('document_frequency.json', sorted_df)

    return sorted_df

def calculate_tfidf(term, freq, document_frequency_dict, num_documents):
    if term in document_frequency_dict:
        tfidf = freq * math.log(num_documents / document_frequency_dict[term], 2)
        return round(tfidf, 2)  # Rundung auf 2 Nachkommastellen
    return 0

def tfidf(term_lists, document_frequency_dict, num_documents):
    with ThreadPoolExecutor() as executor:
        # Parallelisiertes Mapping: Berechnung von (term, tfidf)
        results = executor.map(
            lambda item: (item[0], calculate_tfidf(item[0], item[1], document_frequency_dict, num_documents)),
            term_lists.items()
        )

    # Ergebnisse als Dictionary speichern
    tfidfs = dict(results)

    # Sortieren und die Top-10 auswählen
    sorted_tf_idf = sorted(tfidfs.items(), key=lambda item: item[1], reverse=True)
    return {k: v for k, v in sorted_tf_idf[:10]}

def process_publication_tfidf(publication):
    publication['kombinierte Terme referenced_works'] = tfidf(
        publication['kombinierte Terme referenced_works'], document_frequency_dict, num_documents)
    publication['kombinierte Terme referencing_works'] = tfidf(
        publication['kombinierte Terme referencing_works'], document_frequency_dict, num_documents)
    publication['kombinierte Terme reference_works'] = tfidf(
        publication['kombinierte Terme reference_works'], document_frequency_dict, num_documents)
    publication['kombinierte Terme co_reference_works'] = tfidf(
        publication['kombinierte Terme co_reference_works'], document_frequency_dict, num_documents)
    publication['kombinierte Terme co_referenced_works'] = tfidf(
        publication['kombinierte Terme co_referenced_works'], document_frequency_dict, num_documents)
    publication['kombinierte Terme co_referencing_works'] = tfidf(
        publication['kombinierte Terme co_referencing_works'], document_frequency_dict, num_documents)
    return publication


def assign_tfidf(publications_unique):
    with ThreadPoolExecutor() as executor:
        # Hinzufügen des Fortschrittsbalkens
        publications_unique = list(tqdm(executor.map(process_publication_tfidf, publications_unique),
                                        total=len(publications_unique),
                                        desc="Verarbeiten der Publikationen"))
    return publications_unique

def save_to_json(filename, data):
    """Hilfsfunktion zum Speichern von Daten in eine JSON-Datei"""
    with open(Path(filename), "w") as f:
        json.dump(data, f)

def load_from_json(filename):
    """Hilfsfunktion zum Laden von Daten aus einer JSON-Datei"""
    try:
        with open(Path(filename), "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Warnung: Die Datei {filename} wurde nicht gefunden. Es wird eine leere Liste zurückgegeben.")
        return []
    except json.JSONDecodeError as e:
        print(f"Fehler: Die Datei {filename} konnte nicht geparst werden. Ursache: {str(e)}")
        return []

def process_large_json(filename):
    """Stream-Verarbeitung einer großen JSON-Datei mit festen Metadatenfeldern."""
    result = []  # Liste, um die verarbeiteten Datensätze zu speichern
    try:
        with open(filename, "r") as f:
            # Iteriert über jedes JSON-Objekt in einem Array
            for idx, publication in enumerate(ijson.items(f, "item"), start=1):
                try:
                    # Metadatenfelder extrahieren, mit Standardwerten, falls sie fehlen
                    metadata = {
                        'id': publication['id'],  # Muss vorhanden sein, daher kein .get
                        'title': publication['title'],  # Muss vorhanden sein
                        'authorships': publication['authorships'],  # Muss vorhanden sein
                        'abstract': publication.get("abstract", "") or "",
                        'language': publication.get('language', ""),
                        'kombinierte Terme Titel und Abstract': {},  # Platzhalter für Logik
                        'referenced_works_count': publication.get('referenced_works_count', 0),
                        'referenced_works': publication.get('referenced_works', []),
                        'Abruf referenced_works': 'offen',
                        'cited_by_count': publication.get('cited_by_count', 0),
                        'referencing_works': publication.get('referencing_works', []),
                        'Abruf referencing_works': 'offen',
                        'kombinierte Terme referencing_works': {},  # Platzhalter für Logik
                        'count_reference': "",
                        'reference_works': [],
                        'kombinierte Terme referenced_works': {},  # Platzhalter für Logik
                        'count_co_referenced': "",
                        'co_referenced_works': [],
                        'count_co_referencing': "",
                        'co_referencing_works': [],
                        'count_co_reference': "",
                        'co_reference_works': []
                    }

                    # Fügt den Metadatensatz hinzu
                    result.append(metadata)

                    # Debug-Ausgabe (optional, kann bei Bedarf entfernt werden)
                    print(f"Metadaten für Datensatz {idx}: {metadata}")

                # Behandlung potenzieller Fehler
                except KeyError as key_error:
                    print(f"Fehler bei Datensatz {idx}: Fehlender Schlüssel {key_error}")
                except TypeError as type_error:
                    print(f"Fehler bei Datensatz {idx}: {type_error}. Überprüfen Sie die Struktur: {publication}")
                except Exception as record_error:
                    print(f"Unbekannter Fehler bei Verarbeitung des Datensatzes {idx}: {record_error}")
    except FileNotFoundError:
        print(f"Warnung: Die Datei '{filename}' wurde nicht gefunden.")
    except OSError as os_error:
        print(f"Betriebssystemfehler bei der Datei '{filename}': {str(os_error)}")
    except ijson.JSONError as json_error:
        print(f"Fehler beim Lesen der JSON-Daten in Datei '{filename}': {json_error}")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Allgemeiner Fehler beim Verarbeiten der Datei '{filename}': {str(e)}")
        print(f"Fehlerdetails:\n{error_details}")

    # Rückgabe der Sammlung
    return result

def combine_dictionaries(*dicts):
    ''' Kombiniert mehrere Dicts, wobei Werte für identische Schlüssel summiert werden '''
    combined = {}
    for dictionary in dicts:
        for k, v in dictionary.items():
            if k in combined:
                combined[k] += v  # oder eine andere Geschäftslogik zur Kombination der Werte
            else:
                combined[k] = v
    return combined

def exclude_dict(dict1,dict2):
    return {key:value for key, value in dict1.items() if key not in dict2}

def process_item(item, reference_publications_dict):
    """
    Verarbeitet ein einzelnes Item, aktualisiert es und entfernt nicht gefundene IDs aus den Feldern.
    Gibt das aktualisierte Item sowie die nicht gefundenen IDs als Sets zurück.
    """
    termset_item = item['kombinierte Terme Titel und Abstract']

    # Sets für nicht gefundene IDs
    not_found_referenced = set()
    not_found_referencing = set()
    not_found_co_referenced = set()
    not_found_co_referencing = set()

    # Verarbeitung der referenced_works
    combined_terms_referenced = {}
    valid_referenced_works = []  # Für gültige IDs
    for id_ref in item['referenced_works']:
        if id_ref in reference_publications_dict:
            valid_referenced_works.append(id_ref)  # Behalte gültige ID
            combined_terms_referenced = combine_dictionaries(combined_terms_referenced, reference_publications_dict[id_ref])
        else:
            not_found_referenced.add(id_ref)
            print("Not Found Referenced: ", not_found_referenced)            # Nicht gefunden hinzufügen            # Nicht gefunden hinzufügen
    item['referenced_works'] = valid_referenced_works  # Entferne ungültige IDs
    combined_terms_referenced_excl = exclude_dict(combined_terms_referenced, termset_item)
    item['kombinierte Terme referenced_works'] = dict(sorted(combined_terms_referenced_excl.items(), key=lambda x: x[1], reverse=True))

    # Verarbeitung der referencing_works
    combined_terms_referencing = {}
    valid_referencing_works = []  # Für gültige IDs
    for id_ref in item['referencing_works']:
        if id_ref in reference_publications_dict:
            valid_referencing_works.append(id_ref)  # Behalte gültige ID
            combined_terms_referencing = combine_dictionaries(combined_terms_referencing, reference_publications_dict[id_ref])
        else:
            not_found_referencing.add(id_ref)  # Nicht gefunden hinzufügen
            print("Not Found Referencing: ", not_found_referencing)
    item['referencing_works'] = valid_referencing_works  # Entferne ungültige IDs
    combined_terms_referencing_excl = exclude_dict(combined_terms_referencing, termset_item)
    item['kombinierte Terme referencing_works'] = dict(sorted(combined_terms_referencing_excl.items(), key=lambda x: x[1], reverse=True))

    # Verarbeitung der co_referenced_works
    combined_terms_co_referenced = {}
    valid_co_referenced_works = []  # Für gültige IDs
    for id_ref in item['co_referenced_works']:
        if id_ref in reference_publications_dict:
            valid_co_referenced_works.append(id_ref)  # Behalte gültige ID
            combined_terms_co_referenced = combine_dictionaries(combined_terms_co_referenced, reference_publications_dict[id_ref])
        else:
            not_found_co_referenced.add(id_ref)
            print("Not Found Co-Referenced: ", not_found_co_referenced)            # Nicht gefunden hinzufügen
    item['co_referenced_works'] = valid_co_referenced_works  # Entferne ungültige IDs
    combined_terms_co_referenced_excl = exclude_dict(combined_terms_co_referenced, termset_item)
    item['kombinierte Terme co_referenced_works'] = dict(sorted(combined_terms_co_referenced_excl.items(), key=lambda x: x[1], reverse=True))

    # Verarbeitung der co_referencing_works
    combined_terms_co_referencing = {}
    valid_co_referencing_works = []  # Für gültige IDs
    for id_ref in item['co_referencing_works']:
        if id_ref in reference_publications_dict:
            valid_co_referencing_works.append(id_ref)  # Behalte gültige ID
            combined_terms_co_referencing = combine_dictionaries(combined_terms_co_referencing, reference_publications_dict[id_ref])
        else:
            not_found_co_referencing.add(id_ref)
            print("Not Found Co-Referencing: ", not_found_co_referencing)
    item['co_referencing_works'] = valid_co_referencing_works  # Entferne ungültige IDs
    combined_terms_co_referencing_excl = exclude_dict(combined_terms_co_referencing, termset_item)
    item['kombinierte Terme co_referencing_works'] = dict(sorted(combined_terms_co_referencing_excl.items(), key=lambda x: x[1], reverse=True))

    # Verknüpfung der kombinierten Terme
    item['kombinierte Terme reference_works'] = combine_dictionaries(item['kombinierte Terme referenced_works'], item['kombinierte Terme referencing_works'])
    item['kombinierte Terme co_reference_works'] = combine_dictionaries(item['kombinierte Terme co_referenced_works'], item['kombinierte Terme co_referencing_works'])

    # Rückgabe des aktualisierten Items und der nicht gefundenen IDs
    return item, not_found_referenced, not_found_referencing, not_found_co_referenced, not_found_co_referencing

# Utility: Chunks erstellen
def chunkify(iterable, chunk_size):
    iterator = iter(iterable)
    while chunk := list(islice(iterator, chunk_size)):
        yield chunk


# Hilfsfunktion: Verarbeitet ein einzelnes Item mit Shared Memory
def process_item_with_shared_memory(item, shm_name, size):
    # Shared Memory laden
    shm = shared_memory.SharedMemory(name=shm_name)
    reference_dict = pickle.loads(shm.buf[:size])  # Dictionary aus Shared Memory deserialisieren

    # Verarbeite das Item (muss von der Hauptfunktion bereitgestellt werden)
    result = process_item(item, reference_dict)  # `process_item` ist hier eine weitere Funktion, die definiert werden muss
    return result


# Hauptfunktion: Kombination von Chunk-Verarbeitung & Shared Memory
def enrichment_publications_parallel(base_pub_unique, output_filename="not_found_IDs.json", chunk_size=1000):
    """
    Parallelisierte Funktion zur Verarbeitung von Publikationsdaten.
    Nutzt Shared Memory für große Dictionaries und verarbeitet die Daten in Chunks.
    """
    # Create das große Dictionary für schnellen Zugriff
    reference_publications_dict = {
        publication['id']: publication['kombinierte Terme Titel und Abstract']
        for publication in base_pub_unique
    }

    # Serialisiere das Dictionary und teile es mit Shared Memory
    serialized = pickle.dumps(reference_publications_dict)
    shm = shared_memory.SharedMemory(create=True, size=len(serialized))
    shm.buf[:len(serialized)] = serialized  # Schreibe die serialisierten Daten in Shared Memory

    # Fortschrittsanzeige
    progress_bar = tqdm(total=len(base_pub_unique), desc="Verarbeitung von Publikationen", unit="Publikationen")

    updated_items = []
    not_found_referenced = set()
    not_found_referencing = set()
    not_found_co_referenced = set()
    not_found_co_referencing = set()

    try:
        # Verarbeitung in Chunks
        for chunk in chunkify(base_pub_unique, chunk_size):
            with Pool(processes=cpu_count()) as pool:
                async_results = [
                    pool.apply_async(
                        process_item_with_shared_memory,  # Verarbeitet ein Item mit Shared Memory
                        args=(item, shm.name, len(serialized)),
                    )
                    for item in chunk
                ]
                # Sammle die Ergebnisse
                for res in async_results:
                    result = res.get()
                    item, nf_ref, nf_refing, nf_coref, nf_corefing = result
                    updated_items.append(item)
                    not_found_referenced.update(nf_ref)
                    not_found_referencing.update(nf_refing)
                    not_found_co_referenced.update(nf_coref)
                    not_found_co_referencing.update(nf_corefing)
                    progress_bar.update(1)
    finally:
        # Fortschrittsanzeige beenden und Shared Memory freigeben
        progress_bar.close()
        shm.close()
        shm.unlink()

    # 'Not found'-Daten speichern
    not_found_data = {
        "not_found_referenced": list(not_found_referenced),
        "not_found_referencing": list(not_found_referencing),
        "not_found_co_referenced": list(not_found_co_referenced),
        "not_found_co_referencing": list(not_found_co_referencing),
    }
    with open(output_filename, "w", encoding="utf-8") as json_file:
        json.dump(not_found_data, json_file, ensure_ascii=False, indent=4)

    print(f"'Not found' Daten wurden in {output_filename} gespeichert.")
    return updated_items

def collect_all_publications(publications_list):
    publications_unique = []
    publications_unique_id = []
    for publications in publications_list:
        for item in publications:
            if item['id'] not in publications_unique_id:
                publications_unique.append(item)
                publications_unique_id.append(item['id'])

    return publications_unique

def merge_and_deduplicate(list1, list2):
    combined_list = list1 + list2
    unique_list = list(set(combined_list))
    return unique_list

def solr_ready (base_publications_unique):
    # Behalten nur der benötigten Felder
    necessary_fields = ["id", "title", "abstract", "kombinierte Terme referenced_works", "kombinierte Terme referencing_works",
                        "kombinierte Terme reference_works", "kombinierte Terme co_referenced_works", "kombinierte Terme co_referencing_works",]
    filtered_publications = []

    for publication in base_publications_unique:
        filtered_publication = {field: publication[field] for field in necessary_fields if field in publication}
        filtered_publications.append(filtered_publication)

    return filtered_publications

def consistency_check(base_pub_unique, referenced_pub_unique, referencing_pub_unique, co_referenced_pub_unique, co_referencing_pub_unique):
    unique_referencing_pub_base = []
    for item in base_pub_unique:
        unique_referencing_pub_base = merge_and_deduplicate(unique_referencing_pub_base, item['referencing_works'])

    unique_referenced_pub_referencing = []
    for item in referencing_pub_unique:
        unique_referenced_pub_referencing = merge_and_deduplicate(unique_referenced_pub_referencing, item['referenced_works'])

    unique_referencing_pub_referenced = []
    for item in referenced_pub_unique:
        unique_referencing_pub_referenced = merge_and_deduplicate(unique_referencing_pub_referenced, item['referencing_works'])

    print("Die Summe der 'referenced_works_count' in base_publications_unique ist:", sum(item.get('referenced_works_count', 0) for item in base_pub_unique))
    print("Die Anzahl der Elemente in referenced_publications_unique ist:", len(referenced_pub_unique), "\n")

    print("Die Summe der 'cited_by_count' in base_publications_unique ist:", sum(item.get('cited_by_count', 0) for item in base_pub_unique))
    print("Die Anzahl der uniquen referencing publications in base_publications_unique ist: ", len(unique_referencing_pub_base))
    print("Die Anzahl der Elemente in referencing_publications_unique ist:", len(referencing_pub_unique), "\n")

    print("Die Summe der 'referenced_works_count' in referencing_publications_unique ist:", sum(item.get('referenced_works_count', 0) for item in referencing_pub_unique))
    print("Die Anzahl der uniquen referenced publications in referencing_publications_unique ist: ", len(unique_referenced_pub_referencing))
    print("Die Anzahl der Elemente in co_referenced_publications_unique ist:", len(co_referenced_pub_unique), "\n")
    # differences
    set_a = set(unique_referenced_pub_referencing)
    set_b = set(item['id'] for item in co_referenced_pub_unique)
    diff_set = set_a-set_b
    print('different:')
    print(diff_set)


    print("Die Summe der 'cited_by_count' in referenced_publications_unique ist:", sum(item.get('cited_by_count', 0) for item in referenced_pub_unique))
    print("Die Anzahl der uniquen referencing publications in referenced_publications_unique ist: ", len(unique_referencing_pub_referenced))
    print("Die Anzahl der Elemente in co_referencing_publications_unique ist:", len(co_referencing_pub_unique), "\n")

    set_a = set(unique_referencing_pub_referenced)
    set_b = set(item['id'] for item in co_referencing_pub_unique)
    diff_set = set_a-set_b
    print('different:')
    print(diff_set)

from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Für den Fortschrittsbalken


# Funktion zur Verarbeitung einer einzelnen Publikation
def process_single_publication(pub, id_to_publication, referenced_works_index, referencing_works_index, min_overlap=1):
    pub_id = pub['id']
    pub_referencing_works = set(pub.get('referencing_works', []))  # Publikationen, die pub zitieren
    pub_referenced_works = set(pub.get('referenced_works', []))  # Publikationen, die von pub zitiert werden

    # Ergebnisse
    co_referencing_works = set()  # Ergebnisse für Co-Referencing
    co_referenced_works = set()  # Ergebnisse für Co-Referenced

    # Co-Referencing: Sammle alle Publikationen, die dieselben Werke zitieren
    for work in pub_referenced_works:
        citing_pubs = referenced_works_index.get(work, set())  # Alle Publikationen, die dieses Werk zitieren
        for pub2_id in citing_pubs:
            if pub2_id != pub_id:  # Sich selbst ausschließen
                pub2 = id_to_publication.get(pub2_id, {})
                pub2_referenced_works = set(pub2.get('referenced_works', []))
                # Überprüfe die Überschneidung
                overlap = len(pub_referenced_works & pub2_referenced_works)
                if overlap >= min_overlap:
                    co_referencing_works.add(pub2_id)

    # Co-Referenced: Sammle alle Publikationen, die von denselben Werken zitiert werden
    for work in pub_referencing_works:
        cited_by_pubs = referencing_works_index.get(work,set())  # Alle Publikationen, die von diesem Werk zitiert werden
        for pub2_id in cited_by_pubs:
            if pub2_id != pub_id:  # Sich selbst ausschließen
                pub2 = id_to_publication.get(pub2_id, {})
                pub2_referencing_works = set(pub2.get('referencing_works', []))
                # Überprüfe die Überschneidung
                overlap = len(pub_referencing_works & pub2_referencing_works)
                if overlap >= min_overlap:
                    co_referenced_works.add(pub2_id)

    return pub_id, list(co_referencing_works), list(co_referenced_works)

# Hauptfunktion mit Fortschrittsanzeige
def add_references_parallel_with_progress(combined_publications_unique):
    # 0. Erstelle ein Set aller vorhandenen IDs
    all_ids = {pub['id'] for pub in combined_publications_unique}

    # Bereinige alle referenced_works, um nur IDs zu behalten, die im Set der IDs vorhanden sind
    for pub in combined_publications_unique:
        if 'referenced_works' in pub:
            pub['referenced_works'] = [ref_id for ref_id in pub['referenced_works'] if ref_id in all_ids]

    # Indexe erstellen
    id_to_publication = {pub['id']: pub for pub in combined_publications_unique}
    referenced_works_index = {}

    # 1. Erstelle den referenced_works_index (Zuordnung zitierende Pub -> zitierte Pub)
    for pub in combined_publications_unique:
        for referenced_id in pub.get('referenced_works', []):
            # Referenzen festhalten
            referenced_works_index.setdefault(referenced_id, set()).add(pub['id'])

    # 2. Invertiere referenced_works_index zu referencing_works_index (zitierte Pub -> zitierende Pub)
    referencing_works_index = {}
    for referenced_work, referencing_publications in referenced_works_index.items():
        for referencing_pub_id in referencing_publications:
            referencing_works_index.setdefault(referencing_pub_id, set()).add(referenced_work)

    # 3. Ergänze die referencing_works in den Publikationen
    for pub in combined_publications_unique:
        pub_id = pub['id']
        # Weisen Sie den referencing_works nur zu, wenn es entsprechende Einträge im Index gibt
        pub['referencing_works'] = list(referenced_works_index.get(pub_id, set()))

    # 4. Parallelisieren der Co-Referenzierung und Co-Zitierung
    with Pool(cpu_count()) as pool:
        # Fortschrittsbalken hinzufügen
        results = []
        for result in tqdm(
                pool.starmap(
                    process_single_publication,
                    [(pub, id_to_publication, referenced_works_index, referencing_works_index)
                     for pub in combined_publications_unique]
                ),
                total=len(combined_publications_unique),
                desc="Verarbeite Publikationen",
        ):
            results.append(result)

    # 5. Ergebnisse in die Publikationen zurückschreiben
    for pub_id, co_referencing_works, co_referenced_works in results:
        pub = id_to_publication[pub_id]
        pub['co_referencing_works'] = co_referencing_works
        pub['co_referenced_works'] = co_referenced_works

    return combined_publications_unique

def handle_publication(publication):  # Umbenannte Funktion
    if publication['reference_works'] == []:
        publication['reference_works'] = merge_and_deduplicate(publication['referenced_works'], publication['referencing_works'])
    publication['co_reference_works'] = merge_and_deduplicate(publication['co_referenced_works'], publication['co_referencing_works'])
    publication['count_reference'] = len(publication['reference_works'])
    publication['referenced_works_count'] = len(publication['referenced_works'])
    publication['cited_by_count'] = len(publication['referencing_works'])
    publication['count_co_reference'] = len(publication['co_reference_works'])
    publication['count_co_referenced'] = len(publication['co_referenced_works'])
    publication['count_co_referencing'] = len(publication['co_referencing_works'])
    return publication


def update_metadata(combined_publications_unique):
    total = len(combined_publications_unique)
    progress_bar = tqdm(total=total, desc="Bearbeitungsstand Update Metadaten", unit="Publikation")

    results = []
    with ThreadPoolExecutor() as executor:
        # Starte die Bearbeitung der Publikationen
        future_to_publication = {executor.submit(handle_publication, pub): pub for pub in combined_publications_unique}

        # Ergebnisse schrittweise verarbeiten, Bearbeitungsstand anzeigen
        for future in as_completed(future_to_publication):
            results.append(future.result())  # Füge verarbeitete Publikation hinzu
            progress_bar.update(1)  # Fortschrittsbalken aktualisieren

    progress_bar.close()  # Fortschrittsanzeige beenden
    save_to_json("updated_raw_combined_publications_unique_with_references.json", results)
    return results

def compress_json_data(data, output_file):
    """
    Komprimiert die übergebenen JSON-Daten und speichert sie in einer Datei.

    :param data: JSON-Daten als Eingabe.
    :param output_file: Pfad zur komprimierten Ausgabedatei im GZIP-Format.
    """
    try:
        # Komprimiere die JSON-Daten direkt
        with gzip.open(output_file, 'wt', encoding='utf-8') as outfile:
            json.dump(data, outfile, separators=(',', ':'))  # JSON komprimiert speichern
        print(f"Die JSON-Daten wurden erfolgreich komprimiert und in {output_file} gespeichert.")
    except Exception as e:
        print(f"Fehler bei der Komprimierung: {e}")

# Hauptprogrammfluss
#Abruf der Metadaten Ausgangspublikation, zitierte und zitierende Publikationen
if __name__ == "__main__":
    pager = Works().filter(primary_location={"source.id": "s79460864|s197106261|s4306418959|s2496055428|s4210204422|s4306420562|s4306418323|s4363608773|s4210223861|s817957|s4306418441|s6756005"}).select(
        ["id", "title",  "referenced_works", "abstract_inverted_index", "cited_by_count",
         "referenced_works_count", "language"])

    #pager = Works().filter(primary_topic={"id": "t10286"}).select(["id", "title", "referenced_works", "abstract_inverted_index","referenced_works_count", "cited_by_count"])

    #base_publications_unique = get_by_api(pager, "raw_IR_journals_conferences_base_publications.json")
    base_publications_unique = load_from_json("raw_IR_journals_conferences_base_publications.json")
    referenced_publications_unique = get_referenced_works(base_publications_unique)
    save_to_json(referenced_publications_unique, "raw_IR_journals_conferences_referenced_publications.json")
    referencing_publications_unique = get_referencing_works(base_publications_unique)
    save_to_json(referencing_publications_unique, "raw_IR_journals_conferences_referencing_publications.json")

    co_referenced_publications_unique = get_referenced_works(referencing_publications_unique)
    co_referencing_publications_unique = get_referencing_works(referenced_publications_unique)

    consistency_check(base_publications_unique, referenced_publications_unique, referencing_publications_unique, co_referenced_publications_unique, co_referencing_publications_unique)

    # Initialisieren Sie hier die `combined_publications_unique`-Daten entsprechend und Speicherung der nicht-angereicherten Metadaten
    combined_publications_unique = collect_all_publications([base_publications_unique, referenced_publications_unique, referencing_publications_unique])
    compress_json_data(combined_publications_unique, "raw_IR_journals_conferences_combined_publications.json")

    #combined_publications_unique = process_large_json("W4385569780_test_raw_combined_publications_unique.json") #
    combined_publications_unique = add_references_parallel_with_progress(combined_publications_unique)

    # Texttransformation Metadaten
    combined_publications_unique = indexing_metadata(combined_publications_unique)

   # Anreicherung der Ausgangspublikationen
    combined_publications_unique = enrichment_publications_parallel(combined_publications_unique)
    combined_publications_unique = update_metadata(combined_publications_unique)
    compress_json_data(combined_publications_unique, "processed_IR_journals_conferences_combined_publications.json")


    # Anwendung von tf-idf
    num_documents = len(combined_publications_unique)
    document_frequency_dict = document_frequency(combined_publications_unique, num_documents)
    combined_publications_unique = assign_tfidf(combined_publications_unique)
    compress_json_data(combined_publications_unique, "tfidf_IR_journals_conferences_combined_publications.json")

    # Bereinigung für Solr
    compress_json_data(solr_ready(combined_publications_unique), "IR_journals_conferences_combined_publications.gz")
