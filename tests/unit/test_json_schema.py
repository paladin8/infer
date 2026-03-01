"""Unit tests for the JSON schema to regex compiler."""

from __future__ import annotations

import json

import pytest

from infer.structured.json_schema import json_schema_to_regex, validate_regex_for_json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches(schema: dict, value: object) -> bool:
    """Check if a JSON-encoded value matches the schema regex."""
    pattern = json_schema_to_regex(schema)
    json_str = json.dumps(value, separators=(",", ":"))
    return validate_regex_for_json(pattern, json_str)


def _matches_raw(schema: dict, raw: str) -> bool:
    """Check if a raw string matches the schema regex."""
    pattern = json_schema_to_regex(schema)
    return validate_regex_for_json(pattern, raw)


# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------


class TestStringType:
    def test_basic_string(self) -> None:
        schema = {"type": "string"}
        assert _matches(schema, "hello")
        assert _matches(schema, "")
        assert _matches(schema, "hello world")

    def test_rejects_non_string(self) -> None:
        schema = {"type": "string"}
        assert not _matches_raw(schema, "hello")  # no quotes
        assert not _matches_raw(schema, "42")
        assert not _matches_raw(schema, "true")


class TestNumberType:
    def test_integers(self) -> None:
        schema = {"type": "number"}
        assert _matches_raw(schema, "0")
        assert _matches_raw(schema, "42")
        assert _matches_raw(schema, "-1")

    def test_decimals(self) -> None:
        schema = {"type": "number"}
        assert _matches_raw(schema, "3.14")
        assert _matches_raw(schema, "-0.5")

    def test_scientific(self) -> None:
        schema = {"type": "number"}
        assert _matches_raw(schema, "1e10")
        assert _matches_raw(schema, "1E-5")
        assert _matches_raw(schema, "1.5e+3")

    def test_rejects_invalid(self) -> None:
        schema = {"type": "number"}
        assert not _matches_raw(schema, "01")  # leading zero
        assert not _matches_raw(schema, ".")
        assert not _matches_raw(schema, "e5")
        assert not _matches_raw(schema, '"42"')  # string, not number


class TestIntegerType:
    def test_valid_integers(self) -> None:
        schema = {"type": "integer"}
        assert _matches_raw(schema, "0")
        assert _matches_raw(schema, "42")
        assert _matches_raw(schema, "-1")

    def test_rejects_floats(self) -> None:
        schema = {"type": "integer"}
        assert not _matches_raw(schema, "3.14")
        assert not _matches_raw(schema, "1.0")


class TestBooleanType:
    def test_true_false(self) -> None:
        schema = {"type": "boolean"}
        assert _matches_raw(schema, "true")
        assert _matches_raw(schema, "false")

    def test_rejects_other(self) -> None:
        schema = {"type": "boolean"}
        assert not _matches_raw(schema, "True")
        assert not _matches_raw(schema, "1")
        assert not _matches_raw(schema, '"true"')


class TestNullType:
    def test_null(self) -> None:
        schema = {"type": "null"}
        assert _matches_raw(schema, "null")

    def test_rejects_other(self) -> None:
        schema = {"type": "null"}
        assert not _matches_raw(schema, "None")
        assert not _matches_raw(schema, '""')


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------


class TestSimpleObject:
    def test_single_required_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assert _matches(schema, {"name": "Alice"})

    def test_rejects_missing_required(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assert not _matches(schema, {})


class TestObjectMultipleFields:
    def test_all_required(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
                "name": {"type": "string"},
            },
            "required": ["age", "name"],
        }
        assert _matches(schema, {"age": 30, "name": "Bob"})

    def test_optional_field_present(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
        assert _matches(schema, {"age": 30, "name": "Bob"})

    def test_optional_field_absent(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
        assert _matches(schema, {"name": "Bob"})


class TestNestedObject:
    def test_nested(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                }
            },
            "required": ["address"],
        }
        assert _matches(schema, {"address": {"city": "NYC"}})


class TestEmptyObject:
    def test_empty_properties(self) -> None:
        schema = {"type": "object", "properties": {}}
        assert _matches_raw(schema, "{}")

    def test_empty_no_properties(self) -> None:
        schema = {"type": "object"}
        assert _matches_raw(schema, "{}")


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------


class TestArrayOfStrings:
    def test_basic(self) -> None:
        schema = {"type": "array", "items": {"type": "string"}}
        assert _matches(schema, ["hello", "world"])
        assert _matches(schema, [])
        assert _matches(schema, ["one"])

    def test_rejects_non_string_items(self) -> None:
        schema = {"type": "array", "items": {"type": "string"}}
        assert not _matches_raw(schema, "[1,2,3]")


class TestArrayOfObjects:
    def test_basic(self) -> None:
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            },
        }
        assert _matches(schema, [{"id": 1}, {"id": 2}])
        assert _matches(schema, [])


class TestArrayMinMaxItems:
    def test_min_items(self) -> None:
        schema = {"type": "array", "items": {"type": "integer"}, "minItems": 2}
        assert not _matches_raw(schema, "[1]")
        assert _matches_raw(schema, "[1,2]")
        assert _matches_raw(schema, "[1,2,3]")

    def test_max_items(self) -> None:
        schema = {"type": "array", "items": {"type": "integer"}, "maxItems": 2}
        assert _matches_raw(schema, "[]")
        assert _matches_raw(schema, "[1]")
        assert _matches_raw(schema, "[1,2]")
        assert not _matches_raw(schema, "[1,2,3]")

    def test_exact_items(self) -> None:
        schema = {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        }
        assert not _matches_raw(schema, "[1]")
        assert _matches_raw(schema, "[1,2]")
        assert not _matches_raw(schema, "[1,2,3]")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnumStrings:
    def test_string_enum(self) -> None:
        schema = {"enum": ["red", "green", "blue"]}
        assert _matches(schema, "red")
        assert _matches(schema, "green")
        assert _matches(schema, "blue")
        assert not _matches(schema, "yellow")

    def test_single_value_enum(self) -> None:
        schema = {"enum": ["only"]}
        assert _matches(schema, "only")
        assert not _matches(schema, "other")


class TestEnumMixed:
    def test_mixed_types(self) -> None:
        schema = {"enum": ["yes", 1, None]}
        assert _matches(schema, "yes")
        assert _matches_raw(schema, "1")
        assert _matches_raw(schema, "null")
        assert not _matches(schema, "no")


# ---------------------------------------------------------------------------
# String constraints
# ---------------------------------------------------------------------------


class TestStringMinMaxLength:
    def test_min_length(self) -> None:
        schema = {"type": "string", "minLength": 3}
        assert _matches(schema, "abc")
        assert _matches(schema, "abcd")
        assert not _matches(schema, "ab")

    def test_max_length(self) -> None:
        schema = {"type": "string", "maxLength": 3}
        assert _matches(schema, "")
        assert _matches(schema, "ab")
        assert _matches(schema, "abc")
        assert not _matches(schema, "abcd")

    def test_exact_length(self) -> None:
        schema = {"type": "string", "minLength": 2, "maxLength": 2}
        assert not _matches(schema, "a")
        assert _matches(schema, "ab")
        assert not _matches(schema, "abc")


# ---------------------------------------------------------------------------
# Complex / nested schemas
# ---------------------------------------------------------------------------


class TestNestedArrayInObject:
    def test_array_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["tags"],
        }
        assert _matches(schema, {"tags": ["a", "b"]})
        assert _matches(schema, {"tags": []})


class TestBooleanInObject:
    def test_boolean_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {"active": {"type": "boolean"}},
            "required": ["active"],
        }
        assert _matches(schema, {"active": True})
        assert _matches(schema, {"active": False})


class TestNullableField:
    def test_nullable_via_enum(self) -> None:
        """A field that can be string or null, expressed via enum."""
        schema = {
            "type": "object",
            "properties": {
                "value": {"enum": ["yes", "no", None]},
            },
            "required": ["value"],
        }
        assert _matches(schema, {"value": "yes"})
        assert _matches(schema, {"value": "no"})
        assert _matches_raw(schema, '{"value":null}')


class TestIntegerFieldInObject:
    def test_integer_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        assert _matches(schema, {"count": 42})
        assert _matches(schema, {"count": 0})
        assert _matches(schema, {"count": -1})


class TestDeeplyNested:
    def test_three_levels(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                            },
                            "required": ["value"],
                        }
                    },
                    "required": ["level2"],
                }
            },
            "required": ["level1"],
        }
        assert _matches(schema, {"level1": {"level2": {"value": "deep"}}})


class TestAllTypesObject:
    def test_all_types(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "a_bool": {"type": "boolean"},
                "b_int": {"type": "integer"},
                "c_null": {"type": "null"},
                "d_num": {"type": "number"},
                "e_str": {"type": "string"},
            },
            "required": ["a_bool", "b_int", "c_null", "d_num", "e_str"],
        }
        value = {
            "a_bool": True,
            "b_int": 42,
            "c_null": None,
            "d_num": 3.14,
            "e_str": "hello",
        }
        assert _matches(schema, value)


class TestRequiredVsOptional:
    def test_required_present_optional_absent(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": "string"},
            },
            "required": ["name"],
        }
        assert _matches(schema, {"name": "Alice"})

    def test_both_present(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": "string"},
            },
            "required": ["name"],
        }
        assert _matches(schema, {"name": "Alice", "nickname": "Ali"})


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestSchemaErrors:
    def test_missing_type(self) -> None:
        with pytest.raises(ValueError, match="type"):
            json_schema_to_regex({})

    def test_unsupported_type(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            json_schema_to_regex({"type": "custom"})

    def test_empty_enum(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            json_schema_to_regex({"enum": []})
