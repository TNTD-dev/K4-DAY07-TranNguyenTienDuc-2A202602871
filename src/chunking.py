from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Split AFTER the terminator so ". ", "! ", "? ", ".\n" stay on the sentence.
        pieces = re.split(r"(?<=\. )|(?<=! )|(?<=\? )|(?<=\.\n)", text)
        sentences = [piece.strip() for piece in pieces if piece.strip()]
        if not sentences:
            return []

        chunks: list[str] = []
        size = self.max_sentences_per_chunk
        for start in range(0, len(sentences), size):
            chunks.append(" ".join(sentences[start : start + size]))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _hard_cut(self, current_text: str) -> list[str]:
        size = max(1, self.chunk_size)
        return [current_text[i : i + size] for i in range(0, len(current_text), size)]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return self._hard_cut(current_text)

        separator, rest = remaining_separators[0], remaining_separators[1:]
        if separator == "":
            return self._hard_cut(current_text)

        parts = current_text.split(separator)
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else current + separator + part
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.extend(self._split(current, rest) if len(current) > self.chunk_size else [current])
            if len(part) > self.chunk_size:
                chunks.extend(self._split(part, rest))
                current = ""
            else:
                current = part
        if current:
            chunks.extend(self._split(current, rest) if len(current) > self.chunk_size else [current])
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class HeadingChunker:
    """Split on markdown headings, then recursively split oversized sections.

    Each heading is re-attached to every sub-chunk so later pieces keep section context.
    """

    HEADING_RE = re.compile(r"(?=^#{1,6}\s)", re.MULTILINE)

    def __init__(self, chunk_size: int = 800) -> None:
        self.chunk_size = chunk_size
        self._fallback = RecursiveChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        sections = [part.strip() for part in self.HEADING_RE.split(text) if part.strip()]
        if not sections:
            return self._fallback.chunk(text)
        doc_title = ""
        first_line = sections[0].splitlines()[0].strip()
        if first_line.startswith("# "):
            doc_title = first_line
        chunks: list[str] = []
        for index, section in enumerate(sections):
            if index > 0 and doc_title and not section.startswith(doc_title):
                section = f"{doc_title}\n\n{section}"
            chunks.extend(self._split_section(section))
        return chunks

    def _split_section(self, section: str) -> list[str]:
        if len(section) <= self.chunk_size:
            return [section]
        lines = section.splitlines()
        heading = lines[0] if lines and lines[0].lstrip().startswith("#") else ""
        body = "\n".join(lines[1:]).strip() if heading else section
        if not body:
            return [section]
        pieces = self._fallback.chunk(body)
        if not heading:
            return pieces
        return [f"{heading}\n\n{piece}" for piece in pieces]


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size).chunk(text),
            "by_sentences": SentenceChunker().chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }
        result: dict = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = (sum(len(chunk) for chunk in chunks) / count) if count else 0.0
            result[name] = {"count": count, "avg_length": avg_length, "chunks": chunks}
        return result
