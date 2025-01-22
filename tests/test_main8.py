# File: tests/test_main8.py
import pytest
from main import handle_publication


@pytest.fixture
def sample_publication():
    return {
        "referenced_works": ["7"],  # Enthält "7"
        "reference_works": [],  # Beginnt leer
        "referencing_works": ["3", "4"],
        "co_referenced_works": ["1", "2", "3"],
        "co_reference_works": [],  # Beginnt leer
        "co_referencing_works": ["3", "5", "6"],
    }


def test_handle_publication_empty_reference_works_with_deduplication(sample_publication):
    result = handle_publication(sample_publication)
    assert "reference_works" in result
    assert len(result["reference_works"]) == 3  # Merged and deduplicated (["3", "4", "7"])
    assert sorted(result["reference_works"]) == sorted(["3", "4", "7"])


def test_handle_publication_co_reference_works_with_deduplication(sample_publication):
    result = handle_publication(sample_publication)
    assert "co_reference_works" in result
    assert len(result["co_reference_works"]) == 5  # Merged and deduplicated (["1", "2", "3", "5", "6"])
    assert sorted(result["co_reference_works"]) == sorted(["1", "2", "3", "5", "6"])


def test_handle_publication_counts(sample_publication):
    result = handle_publication(sample_publication)
    assert result["count_reference"] == 3  # Deduplicated reference_works ["3", "4", "7"]
    assert result["referenced_works_count"] == len(sample_publication["referenced_works"])  # Original bleibt gleich
    assert result["cited_by_count"] == len(sample_publication["referencing_works"])  # ["3", "4"]
    assert result["count_co_reference"] == len(result["co_reference_works"])  # ["1", "2", "3", "5", "6"]
    assert result["count_co_referenced"] == len(sample_publication["co_referenced_works"])  # ["1", "2", "3"]
    assert result["count_co_referencing"] == len(sample_publication["co_referencing_works"])  # ["3", "5", "6"]
