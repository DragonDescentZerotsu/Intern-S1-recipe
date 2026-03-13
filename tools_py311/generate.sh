#!/bin/bash
# Generates Python 3.11-compatible copies of tools that use PEP 695 syntax.
# Re-run this script whenever the upstream tools/ files change.
#
# Patches applied:
#   1. PEP 695 type params -> TypeVar (Python 3.12+ -> 3.11 compat)
#   2. pydantic_ai import -> try/except fallback to ValueError

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS_DIR="$(dirname "$SCRIPT_DIR")/tools"

patch_file() {
    local src="$1"
    local dst="$2"
    sed \
        -e 's/^from typing import \(.*\)/from typing import \1, TypeVar/' \
        -e 's/^def _round_output\[T\](value: T) -> T:/T = TypeVar("T")\ndef _round_output(value: T) -> T:/' \
        -e 's/^def _coerce_enum\[E: StrEnum\](value: E | str, enum_cls: type\[E\], error_cls: type\[ModelRetry\]) -> E:/E = TypeVar("E", bound=StrEnum)\ndef _coerce_enum(value, enum_cls: type, error_cls: type):/' \
        -e 's/^from pydantic_ai import ModelRetry$/try:\n    from pydantic_ai import ModelRetry\nexcept ImportError:\n    ModelRetry = ValueError/' \
        "$src" > "$dst"
    echo "Patched: $dst"
}

patch_file "$TOOLS_DIR/RDKit_tools.py" "$SCRIPT_DIR/RDKit_tools.py"
patch_file "$TOOLS_DIR/full_Haydn.py"  "$SCRIPT_DIR/full_Haydn.py"
