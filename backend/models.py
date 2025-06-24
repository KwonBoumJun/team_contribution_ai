from sqlmodel import SQLModel, Field
from typing import Optional

class AnalysisResult(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    word_count: int
    readability: float
    similarity: float
    score: float
