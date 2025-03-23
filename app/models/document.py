from pydantic import BaseModel
from typing import List, Optional

class Document(BaseModel):
    id: str
    name: str
    path: str
    type: str
    content: Optional[str] = None
    
class DocumentQuery(BaseModel):
    query: str
    top_k: int = 5
    
class SearchResult(BaseModel):
    document: Document
    score: float
    
class SearchResponse(BaseModel):
    results: List[SearchResult]
    answer: str
