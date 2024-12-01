# %%
import difflib

import nltk
import pyalex
from pyalex import Works, config
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

def bearbeitung_metadaten(pager, base_publications_unique):
    for page in chain(pager.paginate(per_page=200, n_max=None)):
        for publication in page:
            publication['id'] = publication['id'].replace("https://openalex.org/", "")
            if any(item['id'] == publication['id'] for item in base_publications_unique):
                break
            author_display_names = [authorship["author"]["display_name"] for authorship in publication["authorships"]]
            publication['authorships'] = author_display_names
            publication['abstract'] = publication["abstract"] or ""

            referenced_works_id = [
                ref_id.replace("https://openalex.org/", "")
                for ref_id in publication.get('referenced_works', [])
            ]
            if publication['cited_by_count'] < 2000:
                base_publications_unique.append({
                    'id': publication['id'], 'title': publication['title'], 'authorships': publication['authorships'],
                    'abstract': publication["abstract"] or "",
                    'referenced_works_count': publication['referenced_works_count'], 'referenced_works': referenced_works_id,
                    'cited_by_count': publication['cited_by_count'], 'referencing_works': [],
                    'reference_works': [],
                    'co_referenced_works': [], 'co_referencing_works': [], 'co_reference_works': []
                })

    return base_publications_unique

def get_by_api(pager):
    # all_publications_unique enthält einmalig die Metadaten aller Ausgangspublikationen
    base_publications_unique = bearbeitung_metadaten(pager, [])

    #save_to_json("publications.json", base_publications_unique)
    print(f"Anzahl der Ausgangspublikationen: {len(base_publications_unique)}")

    base_publications_unique.sort(key=lambda x: x.get('id', 0), reverse=True)
    save_to_json("publications.json", base_publications_unique)

    return base_publications_unique


def get_referenced_works(base_publications_unique, filename):
    # referenced_publications_unique enthält einmalig die Metadaten aller zitierten Publikationen der Ausgangspublikationen
    referenced_publications_unique = []
    referenced_publications_unique_ids = []

    for publication in base_publications_unique:
        referenced_publications_unique_ids = merge_and_deduplicate(referenced_publications_unique_ids, publication['referenced_works'])
    print(len(referenced_publications_unique_ids))
    pager_referenced = build_pager(referenced_publications_unique_ids)
    for pager in pager_referenced:
        referenced_publications_unique = bearbeitung_metadaten(pager, referenced_publications_unique)

    referenced_publications_unique.sort(key=lambda x: x.get('id', 0), reverse=True)

    save_to_json(filename, referenced_publications_unique)

    print(f"Anzahl der unique referenced publications: {len(referenced_publications_unique)}")

    return referenced_publications_unique


def build_pager(list_ids):
    pagers = []

    # Durch die ID-Liste in Schritten von 50 iterieren
    for i in range(0, len(list_ids), 50):
        # Erstellen eines ref_id Strings aus den aktuellen 50 IDs (oder weniger, wenn nicht mehr verfügbar)
        ref_id = '|'.join(str(id) for id in list_ids[i:i + 50])

        # Erstellen eines pager-Objekts mit den aktuellen IDs
        pager = Works().filter(ids={"openalex": ref_id}).select([
            "id", "title", "authorships", "referenced_works",
            "abstract_inverted_index", "cited_by_count", "referenced_works_count"])

        # Hinzufügen des pager-Objekts zur pagers-Liste
        pagers.append(pager)

    return pagers


def get_referencing_works(base_publications_unique, filename):
    referencing_publications_unique = []

    for publication in base_publications_unique:
        referencing_publications = []
        pager_referencing = Works().filter(cites=publication['id']).select(["id", "title", "authorships", "referenced_works", "abstract_inverted_index", "cited_by_count","referenced_works_count"])
        referencing_publications_unique = bearbeitung_metadaten(pager_referencing, referencing_publications_unique)
        referencing_publications = bearbeitung_metadaten(pager_referencing, referencing_publications)
        referencing_publications_ids = [pub['id'] for pub in referencing_publications]

        publication['referencing_works'] = referencing_publications_ids
        publication['reference_works'] = merge_and_deduplicate(publication['referenced_works'], publication['referencing_works'])

    referencing_publications_unique.sort(key=lambda x: x.get('id', 0), reverse=True)
    save_to_json(filename, referencing_publications_unique)

    print(f"Anzahl der unique referencing publications: {len(referencing_publications_unique)}")

    return referencing_publications_unique


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

    try:
        # Erkennung der Sprache des Textes
        language_code = detect(text)
        language_name = langcodes.get(language_code).language_name()
    except LangDetectException:
        language_name = 'english'  # Standardmäßig Englisch verwenden

    # Holen der Stopwords für die erkannte Sprache
    stop_words = get_stopwords_for_language(language_name)
    stemmer = get_stemmer_for_language(language_name)

    # Text in Kleinbuchstaben umwandeln
    text = text.lower()

    # Nicht-Wörter entfernen
    text = re.sub(r'[^\w\s]', '', text)

    # Zahlen und Terme mit weniger als 3 Zeichen entfernen
    text = ' '.join([word for word in text.split() if len(word) >= 3 and not word.isdigit()])

    # Wörter splitten, Stopwörter entfernen und stemmen
    words = text.split()
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]

    # Gefilterte Wörter zu einem String zusammenfügen
    #normalized_text = ' '.join(filtered_words)

    return filtered_words


def assign_tf(terms):
    """
    :param text: The input string in which terms need to be counted.
    :return: A dictionary with each unique term in the input string as keys and their respective counts as values.
    """
    # Dictionary zur Speicherung der Zählwerte
    term_count = {}
    totel_number_terms = len(terms)

    for term in terms:
        # Initialisieren des Zählers, falls das Wort noch nicht im Dictionary ist
        if term not in term_count:
            term_count[term] = 0
        # Zählen des Wortes
        term_count[term] += 1

    return term_count

def document_frequency(publications_unique):
    """
    Hilfunktion zur Ermittlung der document frequency aller Terme in "kombinierte Terme Titel und Abstract"
    für die übergebene Liste an Publikationen
    :param publications_unique: Übergebene Liste mit Publikationen
    :param document_frequency_dict:
    :return: document_frequency_list:
    """
    document_frequency_dict = {}

    for item in publications_unique:
        for term in item:
            if term not in document_frequency_dict:
                document_frequency_dict[term] = 1
            else:
                document_frequency_dict[term] += 1

    sorted_df = sorted(document_frequency_dict.items(), key=lambda item: item[0])
    save_to_json('document_frequency.json', sorted_df)

    return document_frequency_dict

def assign_tfidf(term_lists, document_frequency_dict, num_documents):
# Es wird nur die reine Vorkommenshäufigkeit freq genutzt, Abwandlung in relative Vorkommenshäufigkeit noch offen
    tfidfs = {
        term:freq*math.log(num_documents / document_frequency_dict[term])
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

def term_normalisation(list_publications, filename):
    """Hilfsfunktion zur Normalisierung und Zusammenführung aller Terme in Titel und Abstract jeder Publikation.
        Mit Hilfe der Hilfsfunktion count_terms werden die Terme normalisiert, Dubletten entfernt und die Häufigkeit von Termen gezählt
    """
    for publication in list_publications:
        title = publication.get('title', "")
        abstract = publication.get('abstract', "")
        if abstract is None:
            abstract = ""

        combined_text = f"{title} {abstract}"
        # Text normalisieren
        normalized_text = normalize_text(combined_text)
        publication["kombinierte Terme Titel und Abstract"] = assign_tf(normalized_text)
    save_to_json(filename, list_publications)

def save_to_json(filename, data):
    """Hilfsfunktion zum Speichern von Daten in eine JSON-Datei"""
    with open(Path(filename), "w") as f:
        json.dump(data, f)

def combine_dictionaries(dict1, dict2):
    combined_dict = dict1.copy()  # Start with a copy of the first dictionary

    for key, value in dict2.items():
        if key in combined_dict:
            combined_dict[key] += value  # Sum the counts for common keys
        else:
            combined_dict[key] = value  # Add unique keys from the second dictionary

    return combined_dict

def exclude_dict(dict1,dict2):
    return {key:value for key, value in dict1.items() if key not in dict2}

def assign_co_reference(base_publications_unique, referenced_publications_unique, reference):
    for publication in base_publications_unique:
        for pub in referenced_publications_unique:
            if reference == 'referenced_works':
                if reference in pub and publication['id'] in pub[reference]:
                    publication['co_referencing_works'] = merge_and_deduplicate(
                        publication['co_referencing_works'], pub[reference])
            else:
                if reference == 'referencing_works':
                    if reference in pub and publication['id'] in pub[reference]:
                        publication['co_referencing_works'] = merge_and_deduplicate(
                            publication['co_referenced_works'], pub[reference])

    return base_publications_unique

def enrichment_publications(base_publications_unique, reference_publications_unique, combined_publications_unique, reference):
    """
    Jede Ausgangspublikation in base_publications_unique wird ergänzt um Terme aus den referenzierten bzw. referenzierenden
    Publikationen der jeweiligen Ausgangspublikation. Dabei werden nur Terme übernommen, die noch nicht in der Ausgangspublikation
    enthalten sind.
    """
    # Convert referenced_works list to a dictionary for quick lookup
    reference_publications_dict = {publication['id']: publication['kombinierte Terme Titel und Abstract'] for publication in reference_publications_unique}
    combined_publications_dict = {publication['id']: publication['kombinierte Terme Titel und Abstract'] for publication in combined_publications_unique}
    combined_terms = []

    for id in combined_publications_dict:
        combined_terms.append(combined_publications_dict[id])
            # Aggregate terms and store in the item
    document_frequency_overall_dict = document_frequency(combined_terms)

    # Enrich each item in all_items
    for item in base_publications_unique:
        if reference in item:
            combined_terms_referencing = {}
            reference_publications = []
            num_documents = 0
            for id_ref in item[reference]:
                if id_ref in reference_publications_dict:
                # Add the 'kombinierte Terme' from the referenced publication
                    combined_terms_referencing= combine_dictionaries(combined_terms_referencing, reference_publications_dict[id_ref])
                    #offene Aufgabe: Liste mit dict erstellen
                    reference_publications.append(reference_publications_dict[id_ref])
                    num_documents += 1
            # Aggregate terms and store in the item
            document_frequency_dict = document_frequency(reference_publications)
            combined_terms_item_dict = item['kombinierte Terme Titel und Abstract']
            combined_terms_referencing_excl = exclude_dict(combined_terms_referencing, combined_terms_item_dict)
            item['kombinierte Terme ' + reference] = assign_df(combined_terms_referencing_excl, document_frequency_dict)


    save_to_json('publications.json', base_publications_unique)

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
    necessary_fields = ["id", "title", "authorships", "abstract", "kombinierte Terme referenced_works", "kombinierte Terme referencing_works",
                        "kombinierte Terme reference_works"]
    filtered_publications = []

    for publication in base_publications_unique:
        filtered_publication = {field: publication[field] for field in necessary_fields if field in publication}
        filtered_publications.append(filtered_publication)

    return filtered_publications



# Hauptprogrammfluss
#pager = Works().filter(primary_topic={"id": "T13616"}).select(["id", "title", "authorships", "referenced_works", "abstract_inverted_index","referenced_works_count", "cited_by_count"])

pager = Works().filter(ids={"openalex": "W2053522485"}).select(["id", "title", "authorships", "referenced_works", "abstract_inverted_index", "cited_by_count","referenced_works_count"])

# referencing_publications_complete enthält die Metadaten aller zitierenden Publikationen in der Häufigkeit mit der sie die Ausgangspublikationen zitieren
referencing_publications_complete = []

# referenecd_publications_ids_complete enthält die IDs aller zitierten Publikationen in der Häufigkeit mit der sie von den Ausgangspublikationen zitiert werden
referenced_publications_ids_complete = []

# referenced_publications_list enthält alle IDs der zitierten Publikationen der Ausgangspublikationen
referencing_publications_list = []
combined_terms_tfidf = []

all_publications_unique = []

# Beispiel Aufruf der Funktion
#Abruf der Metadaten Ausgangspublikation, zitierte und zitierende Publikationen
base_publications_unique = get_by_api(pager)
referenced_publications_unique = get_referenced_works(base_publications_unique, "referenced_publications_unique.json")
referencing_publications_unique = get_referencing_works(base_publications_unique, "referencing_publications_unique.json")
co_referenced_publications_unique = get_referencing_works(referenced_publications_unique, "co_referenced_publications_unique.json")
co_referencing_publications_unique = get_referenced_works(referencing_publications_unique, "co_referencing_publications_unique.json")


#Zusammenführung der Terme von Titel und Abstrakt jeder Publikation in dem neuen Feld 'kombinierte Terme Titel und Abstract'.
#Vorkommenshäufigkeit der Terme, Entfernen von Mehrfacheinträgen von Termen, lowercasing, Reduktion auf den Wortstamm, entfernen von Zahlen und Termen bestehend aus weniger als 3 Zeichen

term_normalisation(referenced_publications_unique, "referenced_publications_unique.json")
term_normalisation(referencing_publications_unique, "referencing_publications_unique.json")
term_normalisation(base_publications_unique, "publications.json")
term_normalisation(co_referenced_publications_unique, "co_referenced_publications_unique.json")
term_normalisation(co_referencing_publications_unique, "co_referencing_publications_unique.json")

# Zusammenführung aller Publikationen (Ausgangspublikationen, zitierte und zitierende Publikationen) zur Berechnung der Document Frequency der einzelnen Terme
combined_publications_unique = collect_all_publications([base_publications_unique, referenced_publications_unique, referencing_publications_unique, co_referenced_publications_unique, co_referencing_publications_unique])
reference_publications_unique = collect_all_publications([referenced_publications_unique, referencing_publications_unique, co_referenced_publications_unique, co_referencing_publications_unique])

#num_documents = len(combined_publications_unique)

#combined_publications_reference_unique = collect_all_publications([referenced_publications_unique, referencing_publications_unique])

#Anreicherung der Ausgangspublikationen mit den jeweils 10 Termen mit den höchsten idf Werten
base_publications_unique = assign_co_reference(base_publications_unique, referenced_publications_unique, 'referencing_works')
base_publications_unique = assign_co_reference(base_publications_unique, referencing_publications_unique, 'referenced_works')
enrichment_publications(base_publications_unique, referencing_publications_unique, combined_publications_unique, 'referencing_works')
enrichment_publications(base_publications_unique, referenced_publications_unique, combined_publications_unique, 'referenced_works')
enrichment_publications(base_publications_unique, reference_publications_unique, combined_publications_unique, 'reference_works')
#save_to_json("publications.json", solr_ready(base_publications_unique))

#print(f"Anzahl der all unique publications: {len(all_publications_unique)}")

