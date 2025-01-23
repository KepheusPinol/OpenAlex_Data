import pytest
from main import add_references_parallel_with_progress


def test_add_references_parallel_with_progress_empty_input():
    input_data = []
    result = add_references_parallel_with_progress(input_data)
    assert result == []


def test_add_references_parallel_with_progress_no_references():
    input_data = [
        {"id": "1", "referenced_works": []},
        {"id": "2", "referenced_works": []},
    ]
    result = add_references_parallel_with_progress(input_data)
    assert result[0]["referencing_works"] == []
    assert result[1]["referencing_works"] == []


def test_add_references_parallel_with_progress_basic_case():
    input_data = [
        {"id": "1", "referenced_works": ["2"]},
        {"id": "2", "referenced_works": ["3"]},
        {"id": "3", "referenced_works": []},
    ]
    result = add_references_parallel_with_progress(input_data)
    assert result[0]["referencing_works"] == []
    assert set(result[1]["referencing_works"]) == {"1"}
    assert set(result[2]["referencing_works"]) == {"2"}


def test_add_references_parallel_with_progress_circular_references():
    input_data = [
        {"id": "1", "referenced_works": ["2"]},
        {"id": "2", "referenced_works": ["3"]},
        {"id": "3", "referenced_works": ["1"]},
    ]
    result = add_references_parallel_with_progress(input_data)
    assert set(result[0]["referencing_works"]) == {"3"}
    assert set(result[1]["referencing_works"]) == {"1"}
    assert set(result[2]["referencing_works"]) == {"2"}


def test_add_references_parallel_with_progress_self_reference():
    input_data = [
        {"id": "1", "referenced_works": ["1"]},
        {"id": "2", "referenced_works": ["1"]},
    ]
    result = add_references_parallel_with_progress(input_data)
    assert set(result[0]["referencing_works"]) == {"2"}
    assert result[1]["referencing_works"] == []


def test_add_references_parallel_with_progress_nonexistent_references():
    input_data = [
        {"id": "1", "referenced_works": ["2", "unknown"]},
        {"id": "2", "referenced_works": []},
    ]
    result = add_references_parallel_with_progress(input_data)
    assert result[0]["referenced_works"] == ["2"]
    assert result[0]["referencing_works"] == []
    assert set(result[1]["referencing_works"]) == {"1"}
