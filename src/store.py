from __future__ import annotations

from typing import Any, Callable

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = dict(doc.metadata or {})
        if "doc_id" not in metadata:
            metadata["doc_id"] = doc.id
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": self._embedding_fn(doc.content),
            "metadata": metadata,
        }

    def _result_from_record(self, record: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "id": record["id"],
            "content": record["content"],
            "metadata": record["metadata"],
            "score": score,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records or top_k <= 0:
            return []
        query_embedding = self._embedding_fn(query)
        ranked = sorted(
            (
                (compute_similarity(query_embedding, record["embedding"]), record)
                for record in records
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [self._result_from_record(record, score) for score, record in ranked[:top_k]]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        records = [self._make_record(doc) for doc in docs]
        self._store.extend(records)
        if self._use_chroma and self._collection is not None and records:
            self._collection.add(
                ids=[record["id"] for record in records],
                documents=[record["content"] for record in records],
                embeddings=[record["embedding"] for record in records],
                metadatas=[record["metadata"] for record in records],
            )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute cosine similarity of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        records = self._store
        if metadata_filter:
            records = [
                record
                for record in records
                if all(record.get("metadata", {}).get(key) == value for key, value in metadata_filter.items())
            ]
        return self._search_records(query, records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        remaining: list[dict[str, Any]] = []
        removed = False
        for record in self._store:
            metadata_id = record.get("metadata", {}).get("doc_id")
            if record.get("id") == doc_id or metadata_id == doc_id:
                removed = True
                continue
            remaining.append(record)
        self._store = remaining
        if removed and self._use_chroma and self._collection is not None:
            try:
                self._collection.delete(ids=[doc_id])
            except Exception:
                pass
        return removed
