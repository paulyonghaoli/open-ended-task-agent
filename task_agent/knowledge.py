"""
Knowledge graph with TF-IDF scoring and sentence-level snippet extraction.

Design:
  - Term frequency (TF) is computed per document at index time.
  - Inverse document frequency (IDF) is computed across the corpus.
  - At query time, documents are ranked by sum of TF-IDF weights for query terms.
  - Snippets are selected by scoring each sentence against the query and
    returning the top-N most relevant sentences, preserving their original order.

This replaces the original keyword-count approach with a principled retrieval
model equivalent to a lightweight BM25-like ranker without external dependencies.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from .config import MAX_SEARCH_RESULTS, SNIPPET_SENTENCES

# ── Stop-word filter ───────────────────────────────────────────────────────────
_STOP = frozenset(
    """a an the and or but for nor so yet both either neither not only
    as at by in of on to up is are was were be been being have has had
    do does did will would could should may might shall can this that
    these those it its we our they their them us with from into about
    over under between through which who what when where how very just
    also more than such each other any all some such now then there
    here if else after before since while during while because though
    however therefore thus hence moreover furthermore additionally""".split()
)


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOP]


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n\n+", text)
    return [s.strip() for s in raw if len(s.strip()) > 25]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class DocumentNode:
    key: str
    text: str
    keywords: list[str]
    tf: dict[str, float] = field(default_factory=dict)


class KnowledgeGraph:
    """
    Inverted-index retrieval layer with TF-IDF ranking and
    keyword-based document adjacency (graph edges).
    """

    def __init__(self, documents: dict[str, str]) -> None:
        self.documents: Dict[str, DocumentNode] = {}
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self._idf: Dict[str, float] = {}
        self._build(documents)

    # ── Index construction ─────────────────────────────────────────────────────

    def _build(self, documents: dict[str, str]) -> None:
        # Pass 1: compute TF per document and build inverted index
        for key, text in documents.items():
            tokens = _tokenize(text)
            if not tokens:
                continue
            counts = Counter(tokens)
            total = len(tokens)
            tf = {term: count / total for term, count in counts.items()}
            deduped_kw = list(dict.fromkeys(tokens))
            self.documents[key] = DocumentNode(
                key=key, text=text, keywords=deduped_kw, tf=tf
            )
            for term in set(tokens):
                self.index[term].append(key)

        # Pass 2: IDF with Laplace smoothing
        N = max(len(self.documents), 1)
        for term, doc_list in self.index.items():
            df = len(doc_list)
            self._idf[term] = math.log((N + 1) / (df + 1)) + 1.0

        # Pass 3: adjacency via shared vocabulary (graph edges)
        for key, node in self.documents.items():
            neighbors: Counter[str] = Counter()
            for term in node.keywords:
                for other in self.index.get(term, []):
                    if other != key:
                        neighbors[other] += 1
            self.adjacency[key] = [k for k, _ in neighbors.most_common(3)]

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _tfidf(self, query_tokens: list[str], key: str) -> float:
        node = self.documents.get(key)
        if node is None:
            return 0.0
        return sum(
            node.tf.get(t, 0.0) * self._idf.get(t, 0.0) for t in query_tokens
        )

    def _best_snippet(self, text: str, query_tokens: list[str]) -> str:
        sentences = _split_sentences(text)
        if not sentences:
            return text[:500]
        scored: list[tuple[int, float, str]] = []
        query_set = set(query_tokens)
        for idx, sent in enumerate(sentences):
            sent_tokens = set(_tokenize(sent))
            hits = len(query_set & sent_tokens)
            scored.append((idx, hits, sent))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = sorted(scored[: SNIPPET_SENTENCES], key=lambda x: x[0])
        return " ".join(s for _, _, s in top)

    # ── Public API ─────────────────────────────────────────────────────────────

    @classmethod
    def from_documents(cls, documents: dict[str, str]) -> "KnowledgeGraph":
        return cls(documents)

    def search(
        self, query: str, max_results: int = MAX_SEARCH_RESULTS, min_score: float = 0.015
    ) -> list[dict]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = {
            key: self._tfidf(query_tokens, key)
            for key in self.documents
        }
        ranked = sorted(
            ((k, v) for k, v in scores.items() if v >= min_score),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for key, score in ranked[:max_results]:
            node = self.documents[key]
            snippet = self._best_snippet(node.text, query_tokens)
            results.append(
                {
                    "source": key,
                    "summary": snippet,
                    "score": round(score, 4),
                    "related": self.adjacency.get(key, []),
                }
            )
        return results

    def related(self, key: str) -> list[str]:
        return self.adjacency.get(key, [])

    def document_keys(self) -> list[str]:
        return list(self.documents.keys())
