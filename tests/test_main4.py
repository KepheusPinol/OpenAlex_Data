import pytest
from main import add_references


@pytest.fixture
def mock_input_data():
    return [
        {
            "id": "1",
            "referenced_works": ["2", "3"],
            "referencing_works": [],
            "co_referencing_works": [],
            "co_referenced_works": []
        },
        {
            "id": "2",
            "referenced_works": [],
            "referencing_works": [],
            "co_referencing_works": [],
            "co_referenced_works": []
        },
        {
            "id": "3",
            "referenced_works": ["2"],
            "referencing_works": [],
            "co_referencing_works": [],
            "co_referenced_works": []
        }
    ]


def test_add_references_adds_referencing_works(mock_input_data):
    updated_data = add_references(mock_input_data)
    assert updated_data[0]["referencing_works"] == []
    assert updated_data[1]["referencing_works"] == ["3"]
    assert updated_data[2]["referencing_works"] == ["1"]


def test_add_references_adds_coreferencing_works(mock_input_data):
    updated_data = add_references(mock_input_data)
    assert updated_data[0]["co_referencing_works"] == ["3"]
    assert updated_data[1]["co_referencing_works"] == []
    assert updated_data[2]["co_referencing_works"] == ["1"]


def test_add_references_adds_co_referenced_works(mock_input_data):
    updated_data = add_references(mock_input_data)
    assert updated_data[0]["co_referenced_works"] == ["3"]
    assert updated_data[1]["co_referenced_works"] == []
    assert updated_data[2]["co_referenced_works"] == ["1"]


def test_add_references_ignores_missing_referenced_ids(mock_input_data):
    mock_input_data[0]["referenced_works"].append("999")  # Add non-existent reference ID
    updated_data = add_references(mock_input_data)
    assert "999" not in updated_data[0]["referencing_works"]


def test_add_references_no_duplicate_relationships(mock_input_data):
    updated_data = add_references(mock_input_data)
    assert len(updated_data[0]["referencing_works"]) == len(set(updated_data[0]["referencing_works"]))
    assert len(updated_data[0]["co_referencing_works"]) == len(set(updated_data[0]["co_referencing_works"]))
    assert len(updated_data[0]["co_referenced_works"]) == len(set(updated_data[0]["co_referenced_works"]))
