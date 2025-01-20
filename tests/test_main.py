# File: tests/test_main.py

import json
from pathlib import Path

import pytest
from main import add_references


@pytest.fixture
def mock_input_data():
    return [
        {
            "id": "pub1",
            "referenced_works": ["ref1", "ref2"],
            "referencing_works": [],
            "co_referencing_works": [],
            "co_referenced_works": [],
        },
        {
            "id": "pub2",
            "referenced_works": ["ref2", "ref3"],
            "referencing_works": [],
            "co_referencing_works": [],
            "co_referenced_works": [],
        },
        {
            "id": "pub3",
            "referenced_works": ["ref1"],
            "referencing_works": [],
            "co_referencing_works": [],
            "co_referenced_works": [],
        },
    ]


def test_add_references_correctly_adds_referencing_works(mock_input_data, tmp_path):
    output_file = tmp_path / "Testfile.json"
    save_to_json = lambda filename, data: output_file.write_text(json.dumps(data))

    updated_data = add_references(mock_input_data)

    for pub in updated_data:
        assert isinstance(pub["referencing_works"], list)


def test_add_references_handles_no_referenced_works(mock_input_data):
    for publication in mock_input_data:
        publication.pop("referenced_works")
    updated_data = add_references(mock_input_data)

    for pub in updated_data:
        assert pub["referencing_works"] == []


def test_add_references_adds_co_referencing_works(mock_input_data, tmp_path):
    Practical
    checking
