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

            term_dict = initialize_term_dict(normalize_text(f"{publication['title']} {publication['abstract']}"))
            sorted_term_dict = dict(sorted(term_dict.items()))

            #if publication['cited_by_count'] < 2000 :
            base_publications_unique.append({
                    'id': publication['id'], 'title': publication['title'], 'authorships': publication['authorships'],
                    'abstract': publication["abstract"] or "", 'kombinierte Terme Titel und Abstract' : sorted_term_dict,
            'referenced_works_count': publication['referenced_works_count'], 'referenced_works': referenced_works_id,
                    'cited_by_count': publication['cited_by_count'], 'referencing_works': [], 'kombinierte Terme referencing_works' : {},
                    'count_reference': "", 'reference_works': [], 'kombinierte Terme referenced_works' : {},
                    'count_co_referenced' : "", 'co_referenced_works': [],
                    'count_co_referencing' : "",'co_referencing_works': [],
                    'count_co_reference' : "",'co_reference_works': []
                })

    return base_publications_unique

def get_by_api(pager, filename):
    # all_publications_unique enthält einmalig die Metadaten aller Ausgangspublikationen
    base_publications_unique = bearbeitung_metadaten(pager, [])

    base_publications_unique.sort(key=lambda x: x.get('id', 0), reverse=True)
    save_to_json(filename, base_publications_unique)

    print(f"Anzahl der Ausgangspublikationen: {len(base_publications_unique)}")

    return base_publications_unique


def get_referenced_works(base_publications_unique, filename):
    # referenced_publications_unique enthält einmalig die Metadaten aller zitierten Publikationen der Ausgangspublikationen
    unique_referenced_pubs = []
    referenced_publications_unique_ids = []

    for publication in base_publications_unique:
        referenced_publications_unique_ids = merge_and_deduplicate(referenced_publications_unique_ids, publication['referenced_works'])
    pager_referenced = build_pager(referenced_publications_unique_ids)
    for pager in pager_referenced:
        unique_referenced_pubs = bearbeitung_metadaten(pager, unique_referenced_pubs)

    unique_referenced_pubs.sort(key=lambda x: x.get('id', 0), reverse=True)

    save_to_json(filename, unique_referenced_pubs)

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
            "abstract_inverted_index", "cited_by_count", "referenced_works_count"])

        # Hinzufügen des pager-Objekts zur pagers-Liste
        pagers.append(pager)

    return pagers


def get_referencing_works(base_publications_unique, filename, filename_base):
    referencing_pub_unique = []

    for publication in base_publications_unique:
        referencing_publications = []
        pager_referencing = Works().filter(cites=publication['id']).select(["id", "title", "authorships", "referenced_works", "abstract_inverted_index", "cited_by_count","referenced_works_count"])
        referencing_pub_unique = bearbeitung_metadaten(pager_referencing, referencing_pub_unique)
        referencing_publications = bearbeitung_metadaten(pager_referencing, referencing_publications)
        referencing_publications_ids = [pub['id'] for pub in referencing_publications]

        publication['referencing_works'] = referencing_publications_ids
        publication['reference_works'] = merge_and_deduplicate(publication['referenced_works'], publication['referencing_works'])
        publication['count_reference'] = len(publication['reference_works'])

    referencing_pub_unique.sort(key=lambda x: x.get('id', 0), reverse=True)
    save_to_json(filename_base, base_publications_unique)
    save_to_json(filename, referencing_pub_unique)

    print(f"Anzahl der unique referencing publications: {len(referencing_pub_unique)}")

    return referencing_pub_unique


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

    # Nicht-Wörter entfernen außer
    #text = re.sub(r'[^\w\s]', '', text) kann gelöscht werden sofern neue Version funktioniert
    text = re.sub(r'[^\w\s\-@]', '', text)

    # Zahlen und Terme mit weniger als 3 Zeichen entfernen
    text = ' '.join([word for word in text.split() if len(word) >= 3 and not word.isdigit()])

    # Wörter splitten, Stopwörter entfernen und stemmen
    words = text.split()
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
        if term in term_count_dict:
            term_count_dict[term] += 1
        else:
            term_count_dict[term] = 1

    return term_count_dict


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

def save_to_json(filename, data):
    """Hilfsfunktion zum Speichern von Daten in eine JSON-Datei"""
    with open(Path(filename), "w") as f:
        json.dump(data, f)

def load_from_json(filename):
    """Hilfsfunktion zum Laden von Daten aus einer JSON-Datei"""
    with open(Path(filename), "r") as f:
        data = json.load(f)
    return data

def combine_dictionaries(dict1, dict2):
    ''' Kombiniert zwei Dicts, wobei Werte für identische Schlüssel summiert werden '''
    combined = dict1.copy()
    for k, v in dict2.items():
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

def enrichment_publications(base_pub_unique, reference_pub_unique, reference):
    """
    Jede Ausgangspublikation in base_publications_unique wird ergänzt um Terme aus den referenzierten bzw. referenzierenden
    Publikationen der jeweiligen Ausgangspublikation. Dabei werden nur Terme übernommen, die noch nicht in der Ausgangspublikation
    enthalten sind.
    """
    # Convert referenced_works list to a dictionary for quick lookup
    reference_publications_dict = {publication['id']: publication['kombinierte Terme Titel und Abstract'] for publication in reference_pub_unique}
    print(reference_publications_dict)
    #co_referenced_publications_dict = {publication['id']: publication['kombinierte Terme referencing_works'] for publication in reference_publications_unique}
    #co_referencing_publications_dict = {publication['id']: publication['kombinierte Terme referenced_works'] for publication in reference_publications_unique}
    #co_reference_publications_dict = combine_dictionaries(co_referenced_publications_dict, co_referencing_publications_dict)

    # Enrich each item in all_items
    for item in base_pub_unique:
        termset_item = item['kombinierte Terme Titel und Abstract']
        #print("termset_item", termset_item)

        combined_terms_referencing = {}
        for id_ref in item[reference]:
            print("item reference:", item[reference])
            print("id_ref", id_ref)
            if id_ref in reference_publications_dict:
                # Add the 'kombinierte Terme' from the referenced publication
                termset = reference_publications_dict[id_ref]
                combined_terms_referencing = dict(sorted(combine_dictionaries(combined_terms_referencing, termset)))

                print("Termset:", termset)
            else:
                print(f"Key {id_ref} does not exist in reference_publications_dict")

                # combined_terms_referencing = combine_dictionaries(combined_terms_referencing, co_reference_publications_dict[id_ref])
        # Aggregate terms and store in the item
        combined_terms_referencing_excl = exclude_dict(combined_terms_referencing, termset_item)
        item['kombinierte Terme ' + reference] = combined_terms_referencing_excl

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
    print("Die Anzahl der Elemente in referenced_publications_unique ist:", len(referenced_pub_unique))

    print("Die Summe der 'cited_by_count' in base_publications_unique ist:", sum(item.get('cited_by_count', 0) for item in base_pub_unique))
    print("Die Anzahl der uniquen referencing publications in base_publications_unique ist: ", len(unique_referencing_pub_base))
    print("Die Anzahl der Elemente in referencing_publications_unique ist:", len(referencing_pub_unique))

    print("Die Summe der 'referenced_works_count' in referencing_publications_unique ist:", sum(item.get('referenced_works_count', 0) for item in referencing_pub_unique))
    print("Die Anzahl der uniquen referenced publications in referencing_publications_unique ist: ", len(unique_referenced_pub_referencing))
    print("Die Anzahl der Elemente in co_referencing_publications_unique ist:", len(co_referencing_pub_unique))

    print("Die Summe der 'cited_by_count' in referenced_publications_unique ist:", sum(item.get('cited_by_count', 0) for item in referenced_pub_unique))
    print("Die Anzahl der uniquen referencing publications in referenced_publications_unique ist: ", len(unique_referencing_pub_referenced))
    print("Die Anzahl der Elemente in co_referenced_publications_unique ist:", len(co_referenced_pub_unique))

# Hauptprogrammfluss
#pager = Works().filter(primary_topic={"subfield.id": "subfields/3309"}).select(["id", "title", "authorships", "referenced_works", "abstract_inverted_index","referenced_works_count", "cited_by_count"])
#pager = Works().filter(primary_topic={"id": "t10286"}).select(["id", "title", "authorships", "referenced_works", "abstract_inverted_index","referenced_works_count", "cited_by_count"])

pager = Works().filter(ids={"openalex": "W2053522485"}).select(["id", "title", "authorships", "referenced_works", "abstract_inverted_index", "cited_by_count","referenced_works_count"])

# Hauptprogramm
#Abruf der Metadaten Ausgangspublikation, zitierte und zitierende Publikationen

base_publications_unique = get_by_api(pager, "raw_base_publications.json")
referenced_publications_unique = get_referenced_works(base_publications_unique, "raw_referenced_publications_unique.json")
referencing_publications_unique = get_referencing_works(base_publications_unique, "raw_referencing_publications_unique.json", "raw_base_publications.json")
co_referenced_publications_unique = get_referencing_works(referenced_publications_unique, "raw_co_referenced_publications_unique.json", "raw_referenced_publications_unique.json")
co_referencing_publications_unique = get_referenced_works(referencing_publications_unique, "raw_co_referencing_publications_unique.json")

# Zusammenführung aller Publikationen (Ausgangspublikationen, zitierte und zitierende Publikationen) zur Berechnung der Document Frequency der einzelnen Terme
combined_publications_unique = collect_all_publications([base_publications_unique, referenced_publications_unique, referencing_publications_unique, co_referenced_publications_unique, co_referencing_publications_unique], "raw_combined_publications_unique.json")
reference_publications_unique = collect_all_publications([referenced_publications_unique, referencing_publications_unique, co_referenced_publications_unique, co_referencing_publications_unique], "raw_reference_publications_unique.json")

#combined_publications_unique = collect_all_publications([base_publications_unique, referenced_publications_unique, referencing_publications_unique])
#reference_publications_unique = collect_all_publications([referenced_publications_unique, referencing_publications_unique])

base_publications_unique = load_from_json("raw_base_publications.json")
referenced_publications_unique = load_from_json("raw_referenced_publications_unique.json")
referencing_publications_unique = load_from_json("raw_referencing_publications_unique.json")
co_referenced_publications_unique = load_from_json("raw_co_referenced_publications_unique.json")
co_referencing_publications_unique = load_from_json("raw_co_referencing_publications_unique.json")
combined_publications_unique = load_from_json("raw_combined_publications_unique.json")
reference_publications_unique = load_from_json("raw_reference_publications_unique.json")

consistency_check(base_publications_unique, referenced_publications_unique, referencing_publications_unique, co_referenced_publications_unique, co_referencing_publications_unique)
'''
#Anreicherung der Ausgangspublikationen 
base_publications_unique = assign_co_reference(base_publications_unique, referenced_publications_unique, 'referencing_works')
base_publications_unique = assign_co_reference(base_publications_unique, referencing_publications_unique, 'referenced_works')
referencing_publications_unique = enrichment_publications(referencing_publications_unique, co_referenced_publications_unique, 'referenced_works')
referenced_publications_unique = enrichment_publications(referenced_publications_unique, co_referencing_publications_unique, 'referencing_works')
base_publications_unique = enrichment_publications(base_publications_unique, referenced_publications_unique, 'referenced_works')
base_publications_unique = enrichment_publications(base_publications_unique, referencing_publications_unique, 'referencing_works')
base_publications_unique = enrichment_publications(base_publications_unique, reference_publications_unique, 'reference_works')

save_to_json("referencing_publications_unique.json", referencing_publications_unique)
save_to_json("referenced_publications_unique.json", referenced_publications_unique)
save_to_json("co_referenced_publications_unique.json", co_referenced_publications_unique)
save_to_json("co_referencing_publications_unique.json", co_referencing_publications_unique)
save_to_json("base_publications_unique.json", base_publications_unique)

#enrichment_publications(base_publications_unique, co_referencing_publications_unique, 'co_referencing_works')
#save_to_json("publications.json", solr_ready(base_publications_unique))
'''
