import pytest
from main import add_references


@pytest.fixture
def mock_input_data():
    return [
        {"id": "1", "referenced_works": ["2"], "referencing_works": [], "co_referencing_works": [],
         "co_referenced_works": []},
        {"id": "2", "referenced_works": [], "referencing_works": [], "co_referencing_works": [],
         "co_referenced_works": []},
        {"id": "3", "referenced_works": ["1"], "referencing_works": [], "co_referencing_works": [],
         "co_referenced_works": []},
    ]


def test_add_references_adds_referencing_works(mock_input_data):
    updated_data = add_references(mock_input_data)
    assert updated_data[0]["referencing_works"] == ["3"]
    assert updated_data[1]["referencing_works"] == ["1"]
    assert updated_data[2]["referencing_works"] == []


def test_add_references_calculates_co_referencing_works(mock_input_data):
    mock_input_data[0]["referencing_works"] = ["3"]
    updated_data = add_references(mock_input_data)
    assert updated_data[0]["co_referencing_works"] == []
    assert updated_data[1]["co_referencing_works"] == []
    assert updated_data[2]["co_referencing_works"] == ["1"]


def test_add_references_calculates_co_referenced_works(mock_input_data):
    mock_input_data[2]["referenced_works"] = ["2"]
    updated_data = add_references(mock_input_data)
    assert updated_data[0]["co_referenced_works"] == []
    assert updated_data[1]["co_referenced_works"] == []
    assert updated_data[2]["co_referenced_works"] == ["1"]


def test_add_references_ignores_empty_publications():
    mock_data = []
    updated_data = add_references(mock_data)
    assert updated_data == []


def test_add_references_handles_missing_keys():
    mock_data = [{"id": "1"}]
    updated_data = add_references(mock_data)
    assert updated_data[0]["referencing_works"] == []
    assert updated_data[0]["co_referencing_works"] == []
    assert updated_data[0]["co_referenced_works"] == []
