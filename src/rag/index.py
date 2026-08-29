"""Local hybrid retrieval over the review corpus: dense (sentence embeddings)
+ sparse (BM25) search, fused with reciprocal rank fusion.

Reviews are embedded with a small local HuggingFace model (no external API
calls) and indexed with llama_index's in-memory VectorStoreIndex. A BM25
index is built over the same nodes so exact terms (restaurant names, cuisine
words, district names) that embeddings are weak on can still be found -- this
is the same reason search_reviews/list_places already carry hand-rolled
keyword-matching helpers (see src/text_utils.py). The dense index is
persisted to disk so it isn't rebuilt (re-embedded) on every process start;
BM25 is cheap enough to rebuild from the persisted docstore at startup.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

from .. import config
from ..data_loader import Review, load_places, load_reviews
from ..text_utils import district_matches, normalize_district

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# The default node parser splits any Document over ~1000 chars into multiple
# overlapping chunks. Reviews here top out around 4,000 chars, but every
# chunk of a review shares the same review_id and get_review_by_id/verify_quote
# always ground against the *full* review text regardless of which chunk
# matched -- so chunking buys nothing and actively hurts: a semantic match on
# a chunk deep in a long review (e.g. starting at char 2,800) still causes
# search_reviews to show the model only the review's first 500 chars (see
# tools.py), since the returned text comes from the whole-review metadata,
# not the matched chunk. The model then cites a review it can't actually see
# the relevant part of. chunk_size is set well above the longest review in
# this corpus so every review stays a single node.
Settings.node_parser = SentenceSplitter(chunk_size=8192, chunk_overlap=0)

PERSIST_DIR = config.PROJECT_ROOT / "data" / ".index_cache"

# Bumped whenever the index-building logic (metadata fields, chunking, etc.)
# changes, so a stale on-disk cache built under the old schema is rebuilt
# instead of silently reused.
INDEX_SCHEMA_VERSION = "2"

# Reciprocal-rank-fusion constant. Standard default (see the original RRF
# paper) -- large enough that a single retriever's rank-1 result doesn't
# dominate the fused score outright, small enough that rank still matters.
RRF_K = 60


@dataclass
class SearchHit:
    review: Review
    score: float


class ReviewIndex:
    def __init__(self):
        self.index = self._load_or_build()
        nodes = list(self.index.docstore.docs.values())
        # Built once over the full, unfiltered corpus; per-query filtering is
        # done cheaply afterwards by recomputing a weight mask against this
        # same underlying bm25s index (see BM25Retriever's `existing_bm25`
        # path), so we avoid re-tokenizing/re-indexing on every search() call.
        self._bm25_base = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=len(nodes))
        # BM25Retriever keeps its node-metadata corpus as an attribute on
        # *itself*, not on the underlying bm25s.BM25 index -- the
        # `existing_bm25=` constructor path (which we rely on below to reuse
        # the index without rebuilding it) reads `existing_bm25.corpus`, so
        # mirror it onto the bm25s object here or every filtered lookup 404s.
        self._bm25_base.bm25.corpus = self._bm25_base.corpus

    def _load_or_build(self) -> VectorStoreIndex:
        reviews = load_reviews()
        corpus_hash = self._corpus_hash()
        hash_file = PERSIST_DIR / "corpus_hash.txt"

        if PERSIST_DIR.exists() and hash_file.exists():
            if hash_file.read_text().strip() == corpus_hash:
                storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
                return load_index_from_storage(storage_context)

        places = load_places()
        documents = [
            Document(
                text=self._build_document_text(review, places),
                metadata=self._review_metadata(review),
            )
            for review in reviews
        ]
        index = VectorStoreIndex.from_documents(documents)

        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(PERSIST_DIR))
        hash_file.write_text(corpus_hash)

        return index

    @staticmethod
    def _corpus_hash() -> str:
        """Cheap staleness check: hash stable review IDs/text plus the index
        schema version, so changing chunking/metadata logic also invalidates
        the cache even though the underlying reviews haven't changed."""
        reviews = load_reviews()
        payload = "|".join(f"{review.review_id}:{review.text}" for review in reviews)
        payload = f"schema:{INDEX_SCHEMA_VERSION}|{payload}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _review_metadata(review: Review) -> dict[str, object]:
        return {
            "review_id": review.review_id,
            "place_name": review.place_name,
            "rating": review.rating,
            "text": review.text,
            "language": review.language,
            "timestamp": review.timestamp,
            "owner_response": review.owner_response,
            "district": review.district,
            # Accent/case-normalized so a metadata filter can match "Neukölln"
            # against a "neukolln" query without needing a custom filter fn.
            "district_normalized": normalize_district(review.district) if review.district else None,
        }

    @staticmethod
    def _review_from_metadata(metadata: dict[str, object]) -> Review:
        return Review(
            review_id=str(metadata["review_id"]),
            place_name=str(metadata["place_name"]),
            rating=metadata.get("rating"),
            text=str(metadata["text"]),
            language=metadata.get("language"),
            timestamp=metadata.get("timestamp"),
            owner_response=metadata.get("owner_response"),
            district=metadata.get("district"),
        )

    @staticmethod
    def _build_document_text(review: Review, places: dict[str, object]) -> str:
        place = places.get(review.place_name)
        categories = ", ".join(getattr(place, "categories", []) or [])
        description = getattr(place, "description", None) or ""
        parts = [
            f"place: {review.place_name}",
            f"categories: {categories}" if categories else "",
            f"description: {description}" if description else "",
            f"review: {review.text}",
        ]
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _build_filters(place_name: str | None, district: str | None) -> MetadataFilters | None:
        """Real retriever-level filters instead of over-fetching a fixed
        multiple of top_k and discarding non-matches in Python: a true match
        outside that fixed window used to be silently dropped rather than
        found. TEXT_MATCH_INSENSITIVE mirrors the previous one-directional
        substring check for place_name; district matches against the
        pre-normalized field so accents/case don't need a custom filter fn."""
        filters: list[MetadataFilter] = []
        if place_name:
            filters.append(
                MetadataFilter(key="place_name", value=place_name, operator=FilterOperator.TEXT_MATCH_INSENSITIVE)
            )
        if district:
            filters.append(
                MetadataFilter(
                    key="district_normalized",
                    value=normalize_district(district),
                    operator=FilterOperator.TEXT_MATCH_INSENSITIVE,
                )
            )
        if not filters:
            return None
        return MetadataFilters(filters=filters, condition=FilterCondition.AND)

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[NodeWithScore]], top_k: int
    ) -> list[NodeWithScore]:
        """Combine dense + sparse ranked lists into one ranking. Each list
        contributes 1/(RRF_K + rank) per node it contains; a node found by
        both retrievers accumulates both contributions. This sidesteps the
        fact that cosine similarity and BM25 scores aren't on comparable
        scales -- fusion only needs rank, not raw score, from each list."""
        fused_score: dict[str, float] = {}
        best_node: dict[str, NodeWithScore] = {}
        for nodes in ranked_lists:
            for rank, node_with_score in enumerate(nodes):
                node_id = node_with_score.node.node_id
                fused_score[node_id] = fused_score.get(node_id, 0.0) + 1.0 / (RRF_K + rank + 1)
                best_node.setdefault(node_id, node_with_score)

        ordered_ids = sorted(fused_score, key=lambda nid: fused_score[nid], reverse=True)
        return [
            NodeWithScore(node=best_node[nid].node, score=fused_score[nid]) for nid in ordered_ids[:top_k]
        ]

    def search(
        self,
        query: str,
        place_name: str | None = None,
        district: str | None = None,
        top_k: int = 5,
    ) -> list[SearchHit]:
        filters = self._build_filters(place_name, district)
        candidate_k = max(top_k * 4, 20)

        vector_retriever = VectorIndexRetriever(
            index=self.index, similarity_top_k=candidate_k, filters=filters
        )
        vector_nodes = [
            n for n in vector_retriever.retrieve(query) if float(n.score or 0.0) >= config.MIN_SEARCH_SIMILARITY
        ]

        # BM25 has no cosine-similarity-comparable score to threshold against
        # -- an exact keyword hit is inherently high-precision evidence, so
        # every candidate it returns (after filters) is kept and let fusion
        # rank order it against the dense leg.
        bm25_k = min(candidate_k, len(self._bm25_base.corpus))
        bm25_nodes: list[NodeWithScore] = []
        if bm25_k:
            bm25_retriever = BM25Retriever(
                existing_bm25=self._bm25_base.bm25, filters=filters, similarity_top_k=bm25_k
            )
            bm25_nodes = bm25_retriever.retrieve(query)

        fused = self._reciprocal_rank_fusion([vector_nodes, bm25_nodes], top_k=candidate_k)

        hits: list[SearchHit] = []
        for node_with_score in fused:
            review = self._review_from_metadata(node_with_score.node.metadata)

            # Metadata filters already did the real narrowing; this is a
            # cheap bidirectional safety net for the rare case a query is a
            # superset of the stored district name (filters only check
            # query-substring-of-district, not the reverse).
            if district and not district_matches(district, review.district):
                continue

            hits.append(SearchHit(review=review, score=float(node_with_score.score or 0.0)))

            if len(hits) >= top_k:
                break

        return hits


@lru_cache(maxsize=1)
def get_index() -> ReviewIndex:
    return ReviewIndex()
