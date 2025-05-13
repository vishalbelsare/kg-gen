from pydantic import BaseModel, Field
from typing import List, Tuple, Optional


class Graph(BaseModel):
    entities: set[str] = Field(
        ..., description="All entities including additional ones from response"
    )
    edges: set[str] = Field(..., description="All edges")
    relations: set[Tuple[str, str, str]] = Field(
        ..., description="List of (subject, predicate, object) triples"
    )
    entity_clusters: Optional[dict[str, set[str]]] = None
    edge_clusters: Optional[dict[str, set[str]]] = None

    # mapping chunks
    entities_chunk_ids: Optional[dict[str, List[Tuple[str, int]]]] = None
    relations_chunk_ids: Optional[dict[str, List[Tuple[str, int]]]] = None
    edges_chunk_ids: Optional[dict[str, List[Tuple[str, int]]]] = None
