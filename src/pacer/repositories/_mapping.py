"""Helpers for translating domain models into table column values."""

from enum import Enum
from typing import Any

from pydantic import BaseModel


def row_values(model: BaseModel, *, exclude: set[str] | None = None) -> dict[str, Any]:
    """Domain model -> plain column values, without the id. Enums become strings."""
    data = model.model_dump(exclude={"id", *(exclude or set())})
    return {key: value.value if isinstance(value, Enum) else value for key, value in data.items()}
