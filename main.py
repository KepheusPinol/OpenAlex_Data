# %%
import difflib

import nltk
import pyalex
from pyalex import Works, Sources, config
import json
import re
import math
import langcodes
from itertools import chain
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

#dies ist nur in der co-reference Version enthalten
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

def extract_publication_data(pager, base_publications_unique):
    non_english = 0
    for page in chain(pager.paginate(per_page=200, n_max=None)):
        for publication in page:

            publication['id'] = publication['id'].replace("https://openalex.org/", "")
            if any(item['id'] == publication['id'] for item in base_publications_unique):
                continue
            author_display_names = [authorship["author"]["display_name"] for authorship in publication["authorships"]]
            publication['authorships'] = author_display_names
            publication['abstract'] = publication["abstract"] or ""

            referenced_works_id = [
                ref_id.replace("https://openalex.org/", "")
                for ref_id in publication.get('referenced_works', [])
            ]

            #if publication['language'] == "en":
            base_publications_unique.append({
                'id': publication['id'], 'title': publication['title'], 'authorships': publication['authorships'],
                'abstract': publication["abstract"] or "", 'language': publication.get('language', ""), 'kombinierte Terme Titel und Abstract' : {},
                'referenced_works_count': publication['referenced_works_count'], 'referenced_works': referenced_works_id, 'Abruf referenced_works':'offen',
                'cited_by_count': publication['cited_by_count'], 'referencing_works': [], 'Abruf referencing_works':'offen','kombinierte Terme referencing_works' : {},
                'count_reference': "", 'reference_works': [], 'kombinierte Terme referenced_works' : {},
                'count_co_referenced' : "", 'co_referenced_works': [],
                'count_co_referencing' : "",'co_referencing_works': [],
                'count_co_reference' : "",'co_reference_works': []
            })
            #else:
                #non_english = non_english + 1

    #print(f"Anzahl der non-English Publikationen: {non_english}")
    return base_publications_unique

def get_by_api(pager, filename):
    # all_publications_unique enthält einmalig die Metadaten aller Ausgangspublikationen
    base_publications_unique = extract_publication_data(pager, [])

    base_publications_unique.sort(key=lambda x: x.get('id', 0), reverse=True)
    save_to_json(filename, base_publications_unique)

    print(f"Anzahl der Ausgangspublikationen: {len(base_publications_unique)}")

    return base_publications_unique


def get_referenced_works(base_publications_unique, filename):
    # referenced_publications_unique enthält einmalig die Metadaten aller zitierten Publikationen der Ausgangspublikationen
    unique_referenced_pubs = []
    referenced_publications_unique_ids = []
    referenced_works_count = 0

    for publication in base_publications_unique:
        referenced_publications_unique_ids = merge_and_deduplicate(referenced_publications_unique_ids, publication['referenced_works'])
    pager_referenced = build_pager(referenced_publications_unique_ids)

    for pager in pager_referenced:
        unique_referenced_pubs = extract_publication_data(pager, unique_referenced_pubs)
        unique_referenced_pubs.sort(key=lambda x: x.get('id', 0), reverse=True)
        save_to_json(filename, unique_referenced_pubs)
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
            "id", "title", "authorships", "referenced_works",
            "abstract_inverted_index", "cited_by_count", "referenced_works_count", "language"])

        # Hinzufügen des pager-Objekts zur pagers-Liste
        pagers.append(pager)

    return pagers


def get_referencing_works(base_publications_unique, filename, filename_base):
    referencing_pub_unique = []
    count_pub = 0
    referencing_pub_unique = load_from_json(filename)

    for publication in base_publications_unique:
        if publication['Abruf referencing_works'] == 'offen':
            referencing_publications = []
            pager_referencing = Works().filter(cites=publication['id']).select(["id", "title", "authorships", "referenced_works", "abstract_inverted_index", "cited_by_count","referenced_works_count","language"])
            referencing_pub_unique = extract_publication_data(pager_referencing, referencing_pub_unique)
            referencing_publications = extract_publication_data(pager_referencing, referencing_publications)
            referencing_publications_ids = [pub['id'] for pub in referencing_publications]

            publication['referencing_works'] = referencing_publications_ids
            publication['reference_works'] = merge_and_deduplicate(publication['referenced_works'], publication['referencing_works'])
            publication['count_reference'] = len(publication['reference_works'])
            publication['Abruf referencing_works'] = 'abgeschlossen'

            referencing_pub_unique.sort(key=lambda x: x.get('id', 0), reverse=True)
            save_to_json(filename_base, base_publications_unique)
            save_to_json(filename, referencing_pub_unique)

            count_pub += 1
            print("Es wurden " + str(count_pub) + " von " + str(len(base_publications_unique)) + " abgerufen.")
        else:
            count_pub += 1
            print("Es wurden " + str(count_pub) + " von " + str(len(base_publications_unique)) + " abgerufen.")
            continue




    print(f"Anzahl der unique referencing publications: {len(referencing_pub_unique)}")

    return referencing_pub_unique

def texttransformation_metadaten(base_publications_unique):
    for publication in base_publications_unique:
        if publication['language'] == 'en':
            term_dict = initialize_term_dict(normalize_text(f"{publication['title']} {publication['abstract']}"))
            sorted_term_dict = dict(sorted(term_dict.items()))
            publication['kombinierte Terme Titel und Abstract'] = sorted_term_dict
        else:
            publication['kombinierte Terme Titel und Abstract'] = {}

    return base_publications_unique


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


def normalize_text(text):
    if text is None:
        text = ""

    #try:
        # Erkennung der Sprache des Textes
    #    language_code = detect(text)
    #    language_name = langcodes.get(language_code).language_name()
    #except LangDetectException:
    language_name = 'English'  # Standardmäßig Englisch verwenden

    # Holen der Stopwords für die erkannte Sprache
    stop_words = get_stopwords_for_language(language_name)
    stemmer = get_stemmer_for_language(language_name)

    # Text in Kleinbuchstaben umwandeln
    text = text.lower()

    # Nicht-Wörter entfernen außer
    #text = re.sub(r'[^\w\s]', '', text) kann gelöscht werden sofern neue Version funktioniert
    text = re.sub(r'[^\w\s\-@]', '', text)

    # Zahlen und Terme mit weniger als 3 Zeichen entfernen
    #text = ' '.join([word for word in text.split() if len(word) >= 3 and not word.isdigit()])

    # Wörter splitten, Stopwörter entfernen und stemmen
    #words = text.split()
    words = [
        word for word in text.split()
        if len(word) >= 3 and not re.match(r'^[\d\W]', word) and not re.fullmatch(r'[\d\W]+', word)
    ]
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]

    # Gefilterte Wörter zu einem String zusammenfügen
    #normalized_text = ' '.join(filtered_words)

    return filtered_words


def initialize_term_dict(terms):
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


def document_frequency(publications_unique, num_documents):
    """
    Hilfunktion zur Ermittlung der document frequency aller Terme in "kombinierte Terme Titel und Abstract"
    für die übergebene Liste an Publikationen
    :param publications_unique: Übergebene Liste mit Publikationen
    :param document_frequency_dict:
    :return: document_frequency_list:
    """
    document_frequency_dict = {}
    publications_unique_dict = {publication['id']: list(combine_dictionaries(
            publication['kombinierte Terme Titel und Abstract'],
            publication['kombinierte Terme referencing_works'],
            publication['kombinierte Terme referenced_works']
        ).keys()) for publication in publications_unique}

    for publication_id, terms in publications_unique_dict.items():
        for term in terms:
            if term not in document_frequency_dict:
                document_frequency_dict[term] = 1
            else:
                document_frequency_dict[term] += 1

    # Entferne Terme mit Wert 1 oder größer als num_documents / 3
    filtered_df = {
        term: count
        for term, count in document_frequency_dict.items()
        if count != 1 and count <= num_documents / 1
    }

    sorted_df = dict(sorted(filtered_df.items(), key=lambda item: item[1], reverse=True))
    save_to_json('document_frequency.json', sorted_df)

    return sorted_df

def assign_tfidf(term_lists, document_frequency_dict, num_documents):
# Es wird nur die reine Vorkommenshäufigkeit freq genutzt, Abwandlung in relative Vorkommenshäufigkeit noch offen
    tfidfs = {
        term: (freq * math.log(num_documents / document_frequency_dict[term], 2) #log e
               if term in document_frequency_dict else 0)
        for term, freq in term_lists.items()
    }

    sorted_tf_idf = sorted(tfidfs.items(), key=lambda item: item[1], reverse=True)

    return {k:v for (k,v) in sorted_tf_idf[:10]}

def assign_df(term_lists, document_frequency_dict):
    dfs = {
        term:document_frequency_dict[term]
            for term, freq in term_lists.items()
    }

    sorted_df = sorted(dfs.items(), key=lambda item: item[1], reverse=True)

    #return {k:v for (k,v) in sorted_df[:20]}
    return [k for (k,v) in sorted_df[:20]]

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

def assign_co_reference(base_publications_unique, reference_publications_unique, reference):
    if reference == 'referenced_works':
        for publication in base_publications_unique:
            for pub in reference_publications_unique:
                    if reference in pub and publication['id'] in pub[reference]:
                        publication['co_referencing_works'] = merge_and_deduplicate(
                            publication['co_referencing_works'], pub[reference])
        publication['count_co_referencing'] = len(publication['co_referencing_works'])
    else:
        for publication in base_publications_unique:
            for pub in reference_publications_unique:
                    if reference in pub and publication['id'] in pub[reference]:
                        publication['co_referenced_works'] = merge_and_deduplicate(
                            publication['co_referenced_works'], pub[reference])
        publication['count_co_referenced'] = len(publication['co_referenced_works'])

    return base_publications_unique
"""
def enrichment_publications(base_pub_unique, reference_pub_unique, reference):
    
    #Jede Ausgangspublikation in base_publications_unique wird ergänzt um Terme aus den referenzierten bzw. referenzierenden
    #Publikationen der jeweiligen Ausgangspublikation. Dabei werden nur Terme übernommen, die noch nicht in der Ausgangspublikation
    #enthalten sind.
    
    # Convert referenced_works list to a dictionary for quick lookup
    reference_publications_dict = {publication['id']: publication['kombinierte Terme Titel und Abstract'] for publication in reference_pub_unique}
    co_referenced_publications_dict = {publication['id']: publication['kombinierte Terme referencing_works'] for publication in reference_pub_unique}
    co_referencing_publications_dict = {publication['id']: publication['kombinierte Terme referenced_works'] for publication in reference_pub_unique}
    #co_reference_publications_dict = combine_dictionaries(co_referenced_publications_dict, co_referencing_publications_dict)

    # Enrich each item in all_items
    for item in base_pub_unique:
        termset_item = item['kombinierte Terme Titel und Abstract']
        combined_terms_referencing = {}
        for id_ref in item[reference]:
            if id_ref in reference_publications_dict:
                # Add the 'kombinierte Terme' from the referenced publication
                combined_terms_referencing = combine_dictionaries(combined_terms_referencing, reference_publications_dict[id_ref], co_referencing_publications_dict[id_ref], co_referenced_publications_dict[id_ref])
            else:
                print(f"Key {id_ref} does not exist in reference_publications_dict")

        # Aggregate terms and store in the item
        combined_terms_referencing_excl = exclude_dict(combined_terms_referencing, termset_item)
        sorted_terms = dict(sorted(combined_terms_referencing_excl.items(), key=lambda x: x[1], reverse=True))
        item['kombinierte Terme ' + reference] = sorted_terms
        item['kombinierte Terme reference_works'] = combine_dictionaries(item['kombinierte Terme referenced_works'], item['kombinierte Terme referencing_works'])
        #item['Anzahl kombinierte Terme ' + reference] = len(sorted_terms)

    return base_pub_unique
"""
def enrichment_publications(base_pub_unique):
    """
    Jede Ausgangspublikation in base_publications_unique wird ergänzt um Terme aus den referenzierten bzw. referenzierenden
    Publikationen der jeweiligen Ausgangspublikation. Dabei werden nur Terme übernommen, die noch nicht in der Ausgangspublikation
    enthalten sind.
    """
    # Convert referenced_works list to a dictionary for quick lookup
    reference_publications_dict = {publication['id']: publication['kombinierte Terme Titel und Abstract'] for publication in base_pub_unique}

    # Enrich each item in all_items
    for item in base_pub_unique:
        termset_item = item['kombinierte Terme Titel und Abstract']

        combined_terms_referenced = {}
        for id_ref in item['referenced_works']:
            if id_ref in reference_publications_dict:
                # Add the 'kombinierte Terme' from the referenced publication
                combined_terms_referenced = combine_dictionaries(combined_terms_referenced, reference_publications_dict[id_ref])
            else:
                print(f"Key {id_ref} does not exist in reference_publications_dict")
        # Aggregate terms and store in the item
        combined_terms_referenced_excl = exclude_dict(combined_terms_referenced, termset_item)
        sorted_terms = dict(sorted(combined_terms_referenced_excl.items(), key=lambda x: x[1], reverse=True))
        item['kombinierte Terme referenced_works'] = sorted_terms

        combined_terms_referencing = {}
        for id_ref in item['referencing_works']:
            if id_ref in reference_publications_dict:
                # Add the 'kombinierte Terme' from the referenced publication
                combined_terms_referencing = combine_dictionaries(combined_terms_referencing, reference_publications_dict[id_ref])
            else:
                print(f"Key {id_ref} does not exist in reference_publications_dict")
        # Aggregate terms and store in the item
        combined_terms_referencing_excl = exclude_dict(combined_terms_referenced, termset_item)
        sorted_terms = dict(sorted(combined_terms_referencing_excl.items(), key=lambda x: x[1], reverse=True))
        item['kombinierte Terme referencing_works'] = sorted_terms

        combined_terms_co_referenced = {}
        for id_ref in item['co_referenced_works']:
            if id_ref in reference_publications_dict:
                # Add the 'kombinierte Terme' from the referenced publication
                combined_terms_co_referenced = combine_dictionaries(combined_terms_co_referenced, reference_publications_dict[id_ref])
            else:
                print(f"Key {id_ref} does not exist in reference_publications_dict")
        # Aggregate terms and store in the item
        combined_terms_co_referenced_excl = exclude_dict(combined_terms_co_referenced, termset_item)
        sorted_terms = dict(sorted(combined_terms_co_referenced_excl.items(), key=lambda x: x[1], reverse=True))
        item['kombinierte Terme co_referenced_works'] = sorted_terms

        combined_terms_co_referencing = {}
        for id_ref in item['co_referencing_works']:
            if id_ref in reference_publications_dict:
                # Add the 'kombinierte Terme' from the referenced publication
                combined_terms_co_referencing = combine_dictionaries(combined_terms_co_referencing, reference_publications_dict[id_ref])
            else:
                print(f"Key {id_ref} does not exist in reference_publications_dict")
        # Aggregate terms and store in the item
        combined_terms_co_referencing_excl = exclude_dict(combined_terms_co_referencing, termset_item)
        sorted_terms = dict(sorted(combined_terms_referencing_excl.items(), key=lambda x: x[1], reverse=True))
        item['kombinierte Terme co_referencing_works'] = sorted_terms

        item['kombinierte Terme reference_works'] = combine_dictionaries(item['kombinierte Terme referenced_works'], item['kombinierte Terme referencing_works'])
        item['kombinierte Terme co_reference_works'] = combine_dictionaries(item['kombinierte Terme co_referenced_works'], item['kombinierte Terme co_referencing_works'])


    return base_pub_unique
def collect_all_publications(publications_list, filename):
    publications_unique = []
    publications_unique_id = []
    for publications in publications_list:
        for item in publications:
            if item['id'] not in publications_unique_id:
                publications_unique.append(item)
                publications_unique_id.append(item['id'])

    save_to_json(filename, publications_unique)
    return publications_unique

def merge_and_deduplicate(list1, list2):
    combined_list = list1 + list2
    unique_list = list(set(combined_list))
    return unique_list

def solr_ready (base_publications_unique):
    # Behalten nur der benötigten Felder
    necessary_fields = ["id", "title", "authorships", "abstract", "kombinierte Terme referenced_works", "kombinierte Terme referencing_works",
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

def add_references(combined_publications_unique):
    citing_pubs_found = 0
    co_citing_pubs_found = 0
    co_cited_pubs_found = 0

    # 1. Indexe vorbereiten
    id_to_publication = {pub['id']: pub for pub in combined_publications_unique}

    # Referenzierungs-Lookup vorbereiten
    referenced_works_index = {}
    for pub in combined_publications_unique:
        for referenced_id in pub.get('referenced_works', []):  # Werke, auf die sich pub bezieht
            if referenced_id not in referenced_works_index:
                referenced_works_index[referenced_id] = set()
            referenced_works_index[referenced_id].add(pub['id'])

    # Referenzierungs-Lookup vorbereiten
    referencing_works_index = {}
    for pub in combined_publications_unique:
        for referencing_id in pub.get('referencing_works', []):  # Werke, auf die sich pub bezieht
            if referencing_id not in referencing_works_index:
                referencing_works_index[referencing_id] = set()
            referencing_works_index[referencing_id].add(pub['id'])

    # 2. Referenzierungen effizient hinzufügen
    for pub in combined_publications_unique:
        pub_id = pub['id']
        referencing_works = pub.get('referencing_works', [])
        referencing_works_set = set(referencing_works)  # Umwandeln in ein Set (schneller Lookup)

        for referencing_id in referenced_works_index.get(pub_id, []):  # Publikationen, die pub referenzieren
            if referencing_id not in referencing_works_set:
                pub['referencing_works'].append(referencing_id)
                citing_pubs_found += 1

    # 3. Co-zitierende Publikationen mit Sets auffinden
    # Speichere geprüfte Beziehungen, um doppelte Prüfungen zu vermeiden
    checked_pairs = set()  # Set zur Speicherung von geprüften Paaren (pub_id, pub2_id)

    for pub in combined_publications_unique:
        pub_id = pub['id']
        pub_referencing_works = set(pub.get('referencing_works', []))  # Publikationen, auf die pub verweist

        # Überspringe Publikationen ohne Referenzierungen
        if not pub_referencing_works:
            continue

        # Finde weitere Publikationen, die ebenfalls auf diese Referenzen verweisen
        for ref_id in pub_referencing_works:
            referenced_pubs = referencing_works_index.get(ref_id, set())  # Publikationen, die ref_id referenzieren

            for pub2_id in referenced_pubs:
                # Überspringe die gleiche Publikation
                if pub_id == pub2_id:
                    continue

                # Überprüfe, ob diese Beziehung bereits geprüft wurde
                if (pub_id, pub2_id) in checked_pairs or (pub2_id, pub_id) in checked_pairs:
                    continue

                pub2 = id_to_publication[pub2_id]
                pub2_referencing_works = set(pub2.get('referencing_works', []))  # Referenzen von pub2

                # Finde die gemeinsamen referenzierten Werke
                common_refs = pub_referencing_works & pub2_referencing_works  # Schnittmenge

                if common_refs:  # Falls gemeinsame Zitationen existieren
                    # Füge pub2_id als *co-cited work* zu pub hinzu
                    if pub2_id not in pub['co_referencing_works']:
                        pub['co_referencing_works'].append(pub2_id)

                    # Füge pub_id als *co-cited work* zu pub2 hinzu
                    if pub_id not in pub2['co_referencing_works']:
                        pub2['co_referencing_works'].append(pub_id)

                # Markiere die Beziehung als geprüft
                co_citing_pubs_found += 1
                checked_pairs.add((pub_id, pub2_id))

    # 4. Co-zitierte Publikationen effizient hinzufügen (analog zu Co-Referenzierungen)
    checked_pairs = set()  # Set zur Speicherung von geprüften Paaren (pub_id, pub2_id)

    for pub in combined_publications_unique:
        pub_id = pub['id']
        pub_referenced_works = set(pub.get('referenced_works', []))  # Publikationen, auf die pub verweist

        # Überspringe Publikationen ohne Referenzierungen
        if not pub_referenced_works:
            continue

        # Finde weitere Publikationen, die ebenfalls auf diese Referenzen verweisen
        for ref_id in pub_referenced_works:
            referencing_pubs = referenced_works_index.get(ref_id, set())  # Publikationen, die ref_id referenzieren

            for pub2_id in referencing_pubs:
                # Überspringe die gleiche Publikation
                if pub_id == pub2_id:
                    continue

                # Überprüfe, ob diese Beziehung bereits geprüft wurde
                if (pub_id, pub2_id) in checked_pairs or (pub2_id, pub_id) in checked_pairs:
                    continue

                pub2 = id_to_publication[pub2_id]
                pub2_referenced_works = set(pub2.get('referenced_works', []))  # Referenzen von pub2

                # Finde die gemeinsamen referenzierten Werke
                common_refs = pub_referenced_works & pub2_referenced_works  # Schnittmenge

                if common_refs:  # Falls gemeinsame Zitationen existieren
                    # Füge pub2_id als *co-cited work* zu pub hinzu
                    if pub2_id not in pub['co_referenced_works']:
                        pub['co_referenced_works'].append(pub2_id)

                    # Füge pub_id als *co-cited work* zu pub2 hinzu
                    if pub_id not in pub2['co_referenced_works']:
                        pub2['co_referenced_works'].append(pub_id)

                # Markiere die Beziehung als geprüft
                co_cited_pubs_found += 1
                checked_pairs.add((pub_id, pub2_id))

    print("Anzahl der gefundenen zitierenden Publikationen:", citing_pubs_found)
    print("Anzahl der gefundenen co-zitierenden Publikationen:", co_citing_pubs_found)
    print("Anzahl der gefundenen co-zitierten Publikationen:", co_cited_pubs_found)

    # Ergebnisse speichern
    save_to_json("raw_combined_publications_unique_with_references.json", combined_publications_unique)
    return combined_publications_unique

def update_metadata(combined_publications_unique):
    for publication in combined_publications_unique:
        if publication['reference_works'] == []:
            publication['reference_works'] = merge_and_deduplicate(publication['referenced_works'], publication['referencing_works'])
        publication['co_reference_works'] = merge_and_deduplicate(publication['co_referenced_works'], publication['co_referencing_works'])
        publication['count_co_reference'] = len(publication['co_reference_works'])
        publication['count_co_referenced'] = len(publication['co_referenced_works'])
        publication['count_co_referencing'] = len(publication['co_referencing_works'])
    save_to_json("updated_raw_combined_publications_unique_with_references.json", combined_publications_unique)
    return combined_publications_unique


# Hauptprogrammfluss
#pager = Works().filter(primary_topic={"subfield.id": "subfields/3309"}).select(["id", "title", "authorships", "referenced_works", "abstract_inverted_index","referenced_works_count", "cited_by_count"])

#Abruf der Metadaten Ausgangspublikation, zitierte und zitierende Publikationen
"""
pager = Works().filter(primary_topic={"id": "t10286"}).select(["id", "title", "authorships", "referenced_works", "abstract_inverted_index","referenced_works_count", "cited_by_count"])

base_publications_unique = load_from_json("t10286_raw_base_publications.json")

#base_publications_unique = get_by_api(pager, "t10286_raw_base_publications.json")
#referenced_publications_unique = get_referenced_works(base_publications_unique, "t10286_raw_referenced_publications_unique.json")
#referencing_publications_unique = get_referencing_works(base_publications_unique, "t10286_raw_referencing_publications_unique.json", "t10286_raw_base_publications.json")

referenced_publications_unique = load_from_json("t10286_raw_referenced_publications_unique.json")
referencing_publications_unique = load_from_json("t10286_raw_referencing_publications_unique.json")

co_referenced_publications_unique = get_referenced_works(referencing_publications_unique, "t10286_raw_co_referenced_publications_unique.json")
co_referencing_publications_unique = get_referencing_works(referenced_publications_unique, "t10286_raw_co_referencing_publications_unique.json", "t10286_raw_referenced_publications_unique.json")
"""
"""
pager = Works().filter(ids={"openalex": "W2053522485"}).select(["id", "title", "authorships", "referenced_works", "abstract_inverted_index", "cited_by_count","referenced_works_count","language"])

base_publications_unique = get_by_api(pager, "W2053522485_raw_base_publications.json")
referenced_publications_unique = get_referenced_works(base_publications_unique, "W2053522485_raw_referenced_publications_unique.json")
referencing_publications_unique = get_referencing_works(base_publications_unique, "W2053522485_raw_referencing_publications_unique.json", "W2053522485_raw_base_publications.json")

co_referencing_publications_unique = get_referencing_works(referenced_publications_unique, "W2053522485_raw_co_referencing_publications_unique.json", "W2053522485_raw_referenced_publications_unique.json")
co_referenced_publications_unique = get_referenced_works(referencing_publications_unique, "W2053522485_raw_co_referenced_publications_unique.json")

# Zusammenführung aller Publikationen (Ausgangspublikationen, zitierte und zitierende Publikationen) zur Berechnung der Document Frequency der einzelnen Terme
combined_publications_unique = collect_all_publications([base_publications_unique, referenced_publications_unique, referencing_publications_unique, co_referenced_publications_unique, co_referencing_publications_unique], "W2053522485_raw_combined_publications_unique.json")
reference_publications_unique = collect_all_publications([referenced_publications_unique, referencing_publications_unique, co_referenced_publications_unique, co_referencing_publications_unique], "W2053522485_raw_reference_publications_unique.json")

consistency_check(base_publications_unique, referenced_publications_unique, referencing_publications_unique, co_referenced_publications_unique, co_referencing_publications_unique)
"""
"""
pager = Works().filter(primary_location={"source.id": "S4306418959|S197106261|S2496055428"}).select(["id", "title", "authorships", "referenced_works", "abstract_inverted_index", "cited_by_count","referenced_works_count","language"])

base_publications_unique = get_by_api(pager, "S4306418959_raw_base_publications.json")
referenced_publications_unique = get_referenced_works(base_publications_unique, "S4306418959_raw_referenced_publications_unique.json")
referencing_publications_unique = get_referencing_works(base_publications_unique, "S4306418959_raw_referencing_publications_unique.json", "S4306418959_raw_base_publications.json")

base_publications_unique = load_from_json("S4306418959_raw_base_publications.json")
referenced_publications_unique = load_from_json("S4306418959_raw_referenced_publications_unique.json")
referencing_publications_unique = load_from_json("S4306418959_raw_referencing_publications_unique.json")

co_referencing_publications_unique = get_referencing_works(referenced_publications_unique, "S4306418959_raw_co_referencing_publications_unique.json", "S4306418959_raw_referenced_publications_unique.json")
co_referenced_publications_unique = get_referenced_works(referencing_publications_unique, "S4306418959_raw_co_referenced_publications_unique.json")

# Zusammenführung aller Publikationen (Ausgangspublikationen, zitierte und zitierende Publikationen) zur Berechnung der Document Frequency der einzelnen Terme
combined_publications_unique = collect_all_publications([base_publications_unique, referenced_publications_unique, referencing_publications_unique, co_referenced_publications_unique, co_referencing_publications_unique], "S4306418959_raw_combined_publications_unique.json")
reference_publications_unique = collect_all_publications([referenced_publications_unique, referencing_publications_unique, co_referenced_publications_unique, co_referencing_publications_unique], "S4306418959_raw_reference_publications_unique.json")

consistency_check(base_publications_unique, referenced_publications_unique, referencing_publications_unique, co_referenced_publications_unique, co_referencing_publications_unique)
"""

# Speicherung der nicht-angereicherten Metadaten
combined_publications_unique = load_from_json("IR_raw_base_publications.json")


#Texttransformation Metadaten
combined_publications_unique = texttransformation_metadaten(combined_publications_unique)


def mock_input_data():
    return [
        {
            "id": "1",
            "referenced_works": ["2","3","4"],
            "referencing_works": [],
            "co_referencing_works": [],
            "co_referenced_works": []
        },
        {
            "id": "2",
            "referenced_works": [],
            "referencing_works": ["1"],
            "co_referencing_works": [],
            "co_referenced_works": []
        },
        {
            "id": "3",
            "referenced_works": ["2"],
            "referencing_works": ["1"],
            "co_referencing_works": [],
            "co_referenced_works": []
        },
        {
            "id": "4",
            "referenced_works": ["2"],
            "referencing_works": ["1"],
            "co_referencing_works": [],
            "co_referenced_works": []
        }
    ]
test_data = mock_input_data()

added_references_publications_unique = add_references(combined_publications_unique)
update_metadata = update_metadata(added_references_publications_unique)

#Anreicherung der Ausgangspublikationen 
base_publications_unique = enrichment_publications(update_metadata)

#Anwendung von tf-idf
num_documents = len(base_publications_unique)
document_frequency_dict = document_frequency(base_publications_unique, num_documents)

for publications in base_publications_unique:
    publications['kombinierte Terme referenced_works'] = assign_tfidf(publications['kombinierte Terme referenced_works'], document_frequency_dict, num_documents)
    publications['kombinierte Terme referencing_works'] = assign_tfidf(publications['kombinierte Terme referencing_works'], document_frequency_dict, num_documents)
    publications['kombinierte Terme reference_works'] = assign_tfidf(publications['kombinierte Terme reference_works'], document_frequency_dict, num_documents)
    publications['kombinierte Terme co_reference_works'] = assign_tfidf(publications['kombinierte Terme co_reference_works'], document_frequency_dict, num_documents)
    publications['kombinierte Terme co_referenced_works'] = assign_tfidf(publications['kombinierte Terme co_referenced_works'], document_frequency_dict, num_documents)
    publications['kombinierte Terme co_referencing_works'] = assign_tfidf(publications['kombinierte Terme co_referencing_works'], document_frequency_dict, num_documents)

save_to_json("base_publications_unique.json", base_publications_unique)
#save_to_json("publications.json", solr_ready(base_publications_unique))

