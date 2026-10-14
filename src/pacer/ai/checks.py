"""Small schemas used by the connection checks (python -m pacer.ai.check) and the live tests.

Real schemas (MeetingExtraction, TalkSummary) live next to the pipelines that use them.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TextDigest(BaseModel):
    """The answer of the structured_check prompt."""

    model_config = ConfigDict(extra="forbid")

    language: Literal["he"]
    summary: str = Field(min_length=10, max_length=200)
    topics: list[str] = Field(min_length=1, max_length=3)


SAMPLE_TEXT = (
    "השבוע המחזור יצא לשלושה ימי התנדבות בצפון. החניכים עבדו בחקלאות ובשיפוץ מועדון נוער, "
    "ובערבים קיימו שיחות על אחריות אישית ועל עבודה בצוות."
)
