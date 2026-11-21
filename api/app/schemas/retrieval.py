from pydantic import BaseModel, Field

class Source(BaseModel): #class made to show the artifact
    source: str = Field(default="unknown", description="")
    page: int | None = Field(description="")
    score: float = Field(description="")
