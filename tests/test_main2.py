import json
from pathlib import Path

import pytest
from main import add_references


def test_add_references_with_empty_input():
    empty_input = []
    result = add_references(empty_input)
    assert result == []


def test_add_references_correctly_handles_referencing_works(mock_input_data, tmp_path):
    output_file = tmp_path / "test_output.json"
    save_to_json = lambda filename, data: output_file.write_text(json.dumps(data))

    updated_data = add_references(mock_input_data)
    for pub in updated_data:
        assert isinstance(pub.get("referencing_works", []), list)


def test_add_references_correctly_handles_co_referencing_works(mock_input_data, tmp_path):
    output_file = tmp_path / "test_output.json"
    save_to_json = lambda filename, data: output_file.write_text(json.dumps(data))

    updated_data = add_references(mock_input_data)
    for pub in updated_data:
        assert isinstance(pub.get("co_referencing_works", []), list)


def test_add_references_correctly_handles_co_referenced_works(mock_input_data, tmp_path):
    output_file = tmp_path / "test_output.json"
    save_to_json = lambda filename, data: output_file.write_text(json.dumps(data))

    updated_data = add_references(mock_input_data)
    for pub in updated_data:
        assert isinstance(pub.get("co_referenced_works", []), list)


def test_add_references_warns_on_missing_ids(mocker, mock_input_data):
    mock_warning = mocker.patch("builtins.print")
    mock_input_data[0]["referencing_works"] = ["non_existent_id"]
    add_references(mock_input_data)
    mock_warning.assert_called_with(
        "Warnung: pub2_id 'non_existent_id' nicht in id_to_publication gefunden. Überspringe...")
