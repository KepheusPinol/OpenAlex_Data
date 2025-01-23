import pytest
from main import calculate_tfidf


def test_calculate_tfidf_valid_term():
    term = "example"
    freq = 3
    document_frequency_dict = {"example": 5, "test": 10}
    num_documents = 20
    result = calculate_tfidf(term, freq, document_frequency_dict, num_documents)
    assert result == 5.32


def test_calculate_tfidf_term_not_in_dict():
    term = "not_in_dict"
    freq = 2
    document_frequency_dict = {"example": 5, "test": 10}
    num_documents = 15
    result = calculate_tfidf(term, freq, document_frequency_dict, num_documents)
    assert result == 0


def test_calculate_tfidf_zero_frequency():
    term = "example"
    freq = 0
    document_frequency_dict = {"example": 8}
    num_documents = 50
    result = calculate_tfidf(term, freq, document_frequency_dict, num_documents)
    assert result == 0


def test_calculate_tfidf_large_document_count():
    term = "example"
    freq = 1
    document_frequency_dict = {"example": 1}
    num_documents = 1000
    result = calculate_tfidf(term, freq, document_frequency_dict, num_documents)
    assert result == 9.97


def test_calculate_tfidf_single_document():
    term = "example"
    freq = 1
    document_frequency_dict = {"example": 1}
    num_documents = 1
    result = calculate_tfidf(term, freq, document_frequency_dict, num_documents)
    assert result == 0
