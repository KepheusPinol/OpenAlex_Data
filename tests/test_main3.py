import json
from pathlib import Path

import pytest
from main import add_references


@pytest.fixture
def mock_input_data():
    return [
        {
            "id": "1",
            "referenced_works": ["2"],
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
        }
    ]


@pytest.fixture
def mock_input_data_with_links():
    return [
        {
            "id": "1",
            "referenced_works": ["2"],
            "referencing_works": [],
            "co_referencing_works": [],
            "co_referenced_works": []
        },
        {
            "id": "2",
            "referenced_works": ["3"],
            "referencing_works": [],
            "co_referencing_works": [],
            "co_referenced_works": []
        },
        {
            "id": "3",
            "referenced_works": [],
            "referencing_works": [],
            "co_referencing_works": [],
            "co_referenced_works": []
        }
    ]


def save_to_json(filepath, data):
    Path(filepath).write_text(json.dumps(data))


def test_add_references_adds_referencing_works(mock_input_data):
    updated_data = add_references(mock_input_data)
    assert updated_data[0]["referencing_works"] == ["2"]
    assert updated_data[1]["referencing_works"] == []


def test_add_references_correctly_adds_co_referencing_works(mock_input_data_with_links):
    updated_data = add_references(mock_input_data_with_links)
    assert "1" in updated_data[2]["co_referencing_works"]
    assert "3" in updated_data[0]["co_referencing_works"]


def test_add_references_correctly_adds_co_referenced_works(mock_input_data_with_links):
    updated_data = add_references(mock_input_data_with_links)
    assert "1" in updated_data[2]["co_referenced_works"]
    assert "2" in updated_data[0]["co_referenced_works"]


def test_add_references_empty_input():
    input_data = []
    updated_data = add_references(input_data)
    assert updated_data == []


def test_add_references_handles_no_referenced_works(mock_input_data):
    for pub in mock_input_data:
        pub["referenced_works"] = []
    updated_data = add_references(mock_input_data)
    for pub in updated_data:
        assert pub["referencing_works"] == []
        assert pub["co_referencing_works"] == []
        assert pub["co_referenced_works"] == []
