import pytest
from main import get_referenced_works


def test_get_referenced_works_with_empty_list():
    base_publications_unique = []
    result = get_referenced_works(base_publications_unique)
    assert result == []


def test_get_referenced_works_with_single_publication():
    base_publications_unique = [
        {"id": "1", "referenced_works": ["2", "3"]}
    ]
    result = get_referenced_works(base_publications_unique)
    assert len(result) > 0
    assert all(pub["id"] in ["2", "3"] for pub in result)


def test_get_referenced_works_with_multiple_publications():
    base_publications_unique = [
        {"id": "1", "referenced_works": ["2", "3"]},
        {"id": "2", "referenced_works": ["4"]},
        {"id": "3", "referenced_works": ["4", "5"]}
    ]
    result = get_referenced_works(base_publications_unique)
    referenced_ids = [pub["id"] for pub in result]
    assert "2" in referenced_ids
    assert "3" in referenced_ids
    assert "4" in referenced_ids
    assert "5" in referenced_ids


def test_get_referenced_works_with_no_references():
    base_publications_unique = [
        {"id": "1", "referenced_works": []}
    ]
    result = get_referenced_works(base_publications_unique)
    assert result == []


def test_get_referenced_works_with_duplicate_references():
    base_publications_unique = [
        {"id": "1", "referenced_works": ["2", "3"]},
        {"id": "2", "referenced_works": ["2", "3"]}
    ]
    result = get_referenced_works(base_publications_unique)
    referenced_ids = {pub["id"] for pub in result}
    assert len(referenced_ids) == 2
    assert "2" in referenced_ids
    assert "3" in referenced_ids
