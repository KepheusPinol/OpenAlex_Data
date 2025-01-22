import pytest
from main import process_item


def test_process_item_basic_case():
    item = {
        "kombinierte Terme Titel und Abstract": {"term1": 1, "term2": 2},
        "referenced_works": ["id1", "id2", "id3"],
        "referencing_works": ["id4", "id5"],
        "co_referenced_works": ["id6", "id7"],
        "co_referencing_works": ["id8"]
    }
    reference_publications_dict = {
        "id1": {"termA": 2, "termB": 3},
        "id2": {"termA": 1, "termC": 4},
        "id4": {"termX": 5},
        "id6": {"termY": 3},
        "id8": {"termZ": 2}
    }
    updated_item, not_found_referenced, not_found_referencing, not_found_co_referenced, not_found_co_referencing = process_item(
        item, reference_publications_dict)

    assert updated_item["referenced_works"] == ["id1", "id2"]
    assert updated_item["referencing_works"] == ["id4"]
    assert updated_item["co_referenced_works"] == ["id6"]
    assert updated_item["co_referencing_works"] == ["id8"]
    assert "id3" in not_found_referenced
    assert "id5" in not_found_referencing
    assert "id7" in not_found_co_referenced


def test_process_item_term_counts_correct_fields():
    item = {
        "kombinierte Terme Titel und Abstract": {"term1": 1, "term2": 2},
        "referenced_works": ["id1", "id2", "id3"],
        "referencing_works": ["id4", "id5"],
        "co_referenced_works": ["id6", "id7"],
        "co_referencing_works": ["id8"]
    }
    reference_publications_dict = {
        "id1": {"termA": 2, "termB": 3},
        "id2": {"termA": 1, "termC": 4},
        "id4": {"termX": 5},
        "id6": {"termY": 3},
        "id8": {"termZ": 2}
    }
    updated_item, _, _, _, _ = process_item(item, reference_publications_dict)

    # Erwartete Terme für referenced_works
    expected_referenced_terms = {
        "termA": 3,  # 2 (id1) + 1 (id2)
        "termB": 3,  # nur aus id1
        "termC": 4  # nur aus id2
    }
    assert updated_item["kombinierte Terme referenced_works"] == expected_referenced_terms

    # Erwartete Terme für referencing_works
    expected_referencing_terms = {
        "termX": 5  # nur aus id4
    }
    assert updated_item["kombinierte Terme referencing_works"] == expected_referencing_terms

    # Erwartete Terme für co_referenced_works
    expected_co_referenced_terms = {
        "termY": 3  # nur aus id6
    }
    assert updated_item["kombinierte Terme co_referenced_works"] == expected_co_referenced_terms

    # Erwartete Terme für co_referencing_works
    expected_co_referencing_terms = {
        "termZ": 2  # nur aus id8
    }
    assert updated_item["kombinierte Terme co_referencing_works"] == expected_co_referencing_terms


def test_process_item_empty_item():
    item = {
        "kombinierte Terme Titel und Abstract": {},
        "referenced_works": [],
        "referencing_works": [],
        "co_referenced_works": [],
        "co_referencing_works": []
    }
    reference_publications_dict = {}
    updated_item, not_found_referenced, not_found_referencing, not_found_co_referenced, not_found_co_referencing = process_item(
        item, reference_publications_dict)

    assert updated_item["referenced_works"] == []
    assert updated_item["referencing_works"] == []
    assert updated_item["co_referenced_works"] == []
    assert updated_item["co_referencing_works"] == []
    assert not not_found_referenced
    assert not not_found_referencing
    assert not not_found_co_referenced
    assert not not_found_co_referencing


def test_process_item_partial_references_missing():
    item = {
        "kombinierte Terme Titel und Abstract": {"term1": 1},
        "referenced_works": ["id1", "id2"],
        "referencing_works": ["id4"],
        "co_referenced_works": [],
        "co_referencing_works": ["id8", "id9"]
    }
    reference_publications_dict = {
        "id1": {"termA": 3},
        "id8": {"termZ": 5}
    }
    updated_item, not_found_referenced, not_found_referencing, not_found_co_referenced, not_found_co_referencing = process_item(
        item, reference_publications_dict)

    assert updated_item["referenced_works"] == ["id1"]
    assert updated_item["referencing_works"] == []
    assert updated_item["co_referenced_works"] == []
    assert updated_item["co_referencing_works"] == ["id8"]
    assert "id2" in not_found_referenced
    assert "id4" in not_found_referencing
    assert "id9" in not_found_co_referencing


def test_process_item_no_references_match():
    item = {
        "kombinierte Terme Titel und Abstract": {"term1": 1},
        "referenced_works": ["id10"],
        "referencing_works": ["id11"],
        "co_referenced_works": ["id12"],
        "co_referencing_works": ["id13"]
    }
    reference_publications_dict = {}
    updated_item, not_found_referenced, not_found_referencing, not_found_co_referenced, not_found_co_referencing = process_item(
        item, reference_publications_dict)

    assert updated_item["referenced_works"] == []
    assert updated_item["referencing_works"] == []
    assert updated_item["co_referenced_works"] == []
    assert updated_item["co_referencing_works"] == []
    assert "id10" in not_found_referenced
    assert "id11" in not_found_referencing
    assert "id12" in not_found_co_referenced
    assert "id13" in not_found_co_referencing


def test_process_item_large_input():
    item = {
        "kombinierte Terme Titel und Abstract": {"term1": 1},
        "referenced_works": [f"id{i}" for i in range(100)],
        "referencing_works": [f"id{i}" for i in range(100, 200)],
        "co_referenced_works": [f"id{i}" for i in range(200, 300)],
        "co_referencing_works": [f"id{i}" for i in range(300, 400)]
    }
    reference_publications_dict = {f"id{i}": {f"term{i}": i} for i in range(50, 350)}
    updated_item, not_found_referenced, not_found_referencing, not_found_co_referenced, not_found_co_referencing = process_item(
        item, reference_publications_dict)

    assert len(updated_item["referenced_works"]) == 50
    assert len(updated_item["referencing_works"]) == 100
    assert len(updated_item["co_referenced_works"]) == 100
    assert len(updated_item["co_referencing_works"]) == 50
    assert len(not_found_referenced) == 50
    assert len(not_found_referencing) == 50
    assert len(not_found_co_referenced) == 100
    assert len(not_found_co_referencing) == 100
