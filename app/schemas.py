"""Request and response shapes for the API.

Author: Abhishek Grover
"""
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class MessageIn(BaseModel):
    message: Text = Field(
        examples=["WINNER! You have been selected for a £900 prize. Call now to claim."]
    )


class BatchIn(BaseModel):
    messages: list[Text] = Field(min_length=1, max_length=100)


class Signal(BaseModel):
    word: str
    weight: float = Field(description="How strongly this word pushed toward the predicted label.")


class Prediction(BaseModel):
    label: Literal["spam", "ham"]
    spam_probability: float
    threshold: float
    signals: list[Signal]


class BatchItem(BaseModel):
    label: Literal["spam", "ham"]
    spam_probability: float


class BatchOut(BaseModel):
    results: list[BatchItem]
