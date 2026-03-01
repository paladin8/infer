"""JSON schema to regex compiler.

Converts a subset of JSON Schema into regex patterns that can be compiled
into FSMs for structured output generation.

Supported schema features:
- type: "object", "array", "string", "number", "integer", "boolean", "null"
- properties + required (for objects)
- items (for arrays, single schema)
- enum (string and number enums)
- minItems, maxItems for arrays
- minLength, maxLength for strings
- Nested objects and arrays

Not supported:
- $ref, allOf, anyOf, oneOf, additionalProperties, pattern
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# JSON regex building blocks
# ---------------------------------------------------------------------------

# Whitespace between JSON tokens.
_WS = "[ \\t\\n\\r]*"

# JSON string: "..." with backslash escapes.
_JSON_STRING = '"([^"\\\\]|\\\\.)*"'

# JSON number: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
_JSON_NUMBER = "-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?"

# JSON integer: -?(0|[1-9][0-9]*)
_JSON_INTEGER = "-?(0|[1-9][0-9]*)"

# JSON boolean: true|false
_JSON_BOOLEAN = "(true|false)"

# JSON null
_JSON_NULL = "null"


# ---------------------------------------------------------------------------
# Regex escaping
# ---------------------------------------------------------------------------

# Characters that need escaping in our regex dialect.
_REGEX_SPECIAL = set("\\[](){}|*+?.")


def _escape_for_regex(text: str) -> str:
    """Escape a literal string for inclusion in a regex pattern.

    Args:
        text: The literal string to escape.

    Returns:
        Escaped string safe for regex concatenation.
    """
    result: list[str] = []
    for ch in text:
        if ch in _REGEX_SPECIAL:
            result.append("\\" + ch)
        elif ch == "\n":
            result.append("\\n")
        elif ch == "\t":
            result.append("\\t")
        elif ch == "\r":
            result.append("\\r")
        else:
            result.append(ch)
    return "".join(result)


# ---------------------------------------------------------------------------
# Schema compilation
# ---------------------------------------------------------------------------


def _compile_type(schema: dict[str, Any]) -> str:
    """Compile a schema based on its 'type' field.

    Args:
        schema: JSON Schema dictionary.

    Returns:
        Regex pattern string.

    Raises:
        ValueError: If the schema type is unsupported or missing.
    """
    schema_type = schema.get("type")

    if schema_type is None:
        # No type specified — check for enum.
        if "enum" in schema:
            return _compile_enum(schema["enum"])
        raise ValueError(f"Schema must have a 'type' or 'enum' field: {schema}")

    if schema_type == "string":
        return _compile_string(schema)
    elif schema_type == "number":
        return _JSON_NUMBER
    elif schema_type == "integer":
        return _JSON_INTEGER
    elif schema_type == "boolean":
        return _JSON_BOOLEAN
    elif schema_type == "null":
        return _JSON_NULL
    elif schema_type == "object":
        return _compile_object(schema)
    elif schema_type == "array":
        return _compile_array(schema)
    else:
        raise ValueError(f"Unsupported schema type: {schema_type!r}")


def _compile_string(schema: dict[str, Any]) -> str:
    """Compile a string schema, respecting minLength/maxLength.

    Args:
        schema: JSON Schema with type "string".

    Returns:
        Regex pattern for JSON strings.
    """
    min_len = schema.get("minLength")
    max_len = schema.get("maxLength")

    if min_len is not None or max_len is not None:
        # Constrain the string content length (excluding quotes).
        # Each character is either a non-quote/non-backslash char or an escape sequence.
        char_pattern = '([^"\\\\]|\\\\.)'
        min_l = min_len if min_len is not None else 0
        quantifier = f"{{{min_l},{max_len}}}" if max_len is not None else f"{{{min_l},}}"
        return f'"{char_pattern}{quantifier}"'

    return _JSON_STRING


def _compile_enum(values: list[Any]) -> str:
    """Compile an enum constraint.

    Args:
        values: List of allowed enum values.

    Returns:
        Regex pattern matching any of the values.

    Raises:
        ValueError: If values list is empty.
    """
    if not values:
        raise ValueError("enum values must not be empty")

    alternatives: list[str] = []
    for v in values:
        if isinstance(v, str):
            # JSON-encode the string (wrap in quotes, escape internals).
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            alternatives.append('"' + _escape_for_regex(escaped) + '"')
        elif isinstance(v, bool):
            # Must check before int since bool is subclass of int.
            alternatives.append("true" if v else "false")
        elif isinstance(v, (int, float)):
            alternatives.append(_escape_for_regex(str(v)))
        elif v is None:
            alternatives.append("null")
        else:
            raise ValueError(f"Unsupported enum value type: {type(v).__name__}")

    if len(alternatives) == 1:
        return alternatives[0]
    return "(" + "|".join(alternatives) + ")"


def _compile_object(schema: dict[str, Any]) -> str:
    """Compile an object schema with properties and required fields.

    Generates a regex that matches JSON objects with the specified fields.
    Required fields must appear; optional fields may be absent.
    Field order is fixed (sorted by name) to keep the regex tractable.

    Args:
        schema: JSON Schema with type "object".

    Returns:
        Regex pattern for matching JSON objects.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    if not properties:
        # Empty object: {}
        return "\\{" + _WS + "\\}"

    # Sort field names for deterministic order.
    field_names = sorted(properties.keys())

    # Build field patterns.
    field_patterns: list[tuple[str, str, bool]] = []
    for name in field_names:
        prop_schema = properties[name]
        value_regex = _compile_type(prop_schema)
        is_required = name in required
        field_patterns.append((name, value_regex, is_required))

    # Build the regex for the object body.
    # We use a fixed field order. Required fields must appear,
    # optional fields are wrapped in (...)?
    # Fields are separated by commas.
    return _build_object_regex(field_patterns)


def _build_object_regex(
    fields: list[tuple[str, str, bool]],
) -> str:
    """Build a regex for a JSON object with fixed field order.

    This handles the complex combinatorics of optional fields with commas.
    We generate a pattern that allows any subset of optional fields to
    appear (in sorted order) with proper comma separation.

    Args:
        fields: List of (name, value_regex, is_required) tuples in sorted order.

    Returns:
        Regex pattern for the object.
    """
    # Strategy: generate all valid field combinations and join with alternation.
    # For small numbers of optional fields this is tractable.
    # For large numbers we'd need a different approach, but educational schemas
    # typically have few fields.

    required_fields = [(n, v) for n, v, r in fields if r]
    optional_fields = [(n, v) for n, v, r in fields if not r]

    # If no optional fields, it's straightforward.
    if not optional_fields:
        return _build_fixed_object_regex(required_fields)

    # Generate all subsets of optional fields.
    # For each subset, combine with required fields in sorted order.
    n_opt = len(optional_fields)

    # Cap at 8 optional fields (256 subsets) to avoid regex explosion.
    if n_opt > 8:
        raise ValueError(f"Too many optional fields ({n_opt}). Maximum supported: 8.")

    alternatives: list[str] = []
    for mask in range(1 << n_opt):
        chosen_opt = [optional_fields[i] for i in range(n_opt) if mask & (1 << i)]
        all_fields = required_fields + chosen_opt
        # Sort by name to maintain deterministic order.
        all_fields.sort(key=lambda x: x[0])
        alternatives.append(_build_fixed_object_regex(all_fields))

    if len(alternatives) == 1:
        return alternatives[0]
    return "(" + "|".join(alternatives) + ")"


def _build_fixed_object_regex(fields: list[tuple[str, str]]) -> str:
    """Build a regex for a JSON object with exactly these fields in order.

    Args:
        fields: List of (name, value_regex) tuples in desired order.

    Returns:
        Regex pattern for the object.
    """
    if not fields:
        return "\\{" + _WS + "\\}"

    parts: list[str] = []
    for i, (name, value_regex) in enumerate(fields):
        key_regex = '"' + _escape_for_regex(name) + '"'
        field_regex = _WS + key_regex + _WS + ":" + _WS + value_regex
        if i > 0:
            field_regex = _WS + "," + field_regex
        parts.append(field_regex)

    return "\\{" + "".join(parts) + _WS + "\\}"


def _compile_array(schema: dict[str, Any]) -> str:
    """Compile an array schema.

    Args:
        schema: JSON Schema with type "array".

    Returns:
        Regex pattern for matching JSON arrays.
    """
    items_schema = schema.get("items", {})
    min_items = schema.get("minItems", 0)
    max_items = schema.get("maxItems")

    # Compile the item type.
    if items_schema:
        item_regex = _compile_type(items_schema)
    else:
        # Any JSON value: simplified to common types.
        item_regex = f"({_JSON_STRING}|{_JSON_NUMBER}|{_JSON_BOOLEAN}|{_JSON_NULL})"

    if min_items == 0 and max_items is None:
        # Zero or more items: [] or [item, item, ...]
        inner = _WS + item_regex + "(" + _WS + "," + _WS + item_regex + ")*"
        return "\\[" + _WS + "(" + inner + _WS + ")?\\]"

    if min_items == 0 and max_items == 0:
        return "\\[" + _WS + "\\]"

    # Build with explicit repetition.
    # Required items (min_items).
    if min_items > 0:
        first = _WS + item_regex
        rest = ("(" + _WS + "," + _WS + item_regex + ")") * (min_items - 1)
        required_part = first + rest
    else:
        required_part = ""

    extra = max_items - min_items if max_items is not None else None

    if extra is not None and extra == 0:
        # Exact count.
        if min_items == 0:
            return "\\[" + _WS + "\\]"
        return "\\[" + required_part + _WS + "\\]"

    # Optional additional items.
    optional_item = "(" + _WS + "," + _WS + item_regex + ")"

    if min_items == 0:
        # All items optional with max.
        if extra is not None:
            # [item?] or [item(,item){0,extra-1}]?
            first_opt = _WS + item_regex
            rest_opt = optional_item + f"{{0,{extra - 1}}}" if extra > 1 else ""
            inner = first_opt + rest_opt
            return "\\[" + _WS + "(" + inner + _WS + ")?\\]"
        else:
            # Zero or more, unlimited.
            inner = _WS + item_regex + "(" + _WS + "," + _WS + item_regex + ")*"
            return "\\[" + _WS + "(" + inner + _WS + ")?\\]"

    optional_part = optional_item + f"{{0,{extra}}}" if extra is not None else optional_item + "*"

    return "\\[" + required_part + optional_part + _WS + "\\]"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def json_schema_to_regex(schema: dict[str, Any]) -> str:
    """Compile a JSON schema to a regex pattern.

    The resulting regex can be compiled into an FSM for constrained generation.

    Supported schema features:
    - type: "object", "array", "string", "number", "integer", "boolean", "null"
    - properties + required (for objects)
    - items (for arrays, single schema)
    - enum (string and number enums)
    - minItems, maxItems for arrays
    - minLength, maxLength for strings
    - Nested objects and arrays

    Args:
        schema: A JSON Schema dictionary.

    Returns:
        A regex pattern string that matches valid JSON conforming to the schema.

    Raises:
        ValueError: If the schema uses unsupported features.
    """
    return _compile_type(schema)


def validate_regex_for_json(pattern: str, json_string: str) -> bool:
    """Test whether a JSON string matches a compiled schema regex.

    This uses Python's re module for validation (not the FSM).
    Useful for testing the regex compiler.

    Args:
        pattern: Regex pattern from json_schema_to_regex.
        json_string: JSON string to validate.

    Returns:
        True if the JSON string matches the pattern.
    """
    return re.fullmatch(pattern, json_string) is not None
