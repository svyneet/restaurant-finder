"""In-process tool functions for the researcher agent: review retrieval +
analytics and free OpenStreetMap-based location lookups. Registered directly
on the pydantic-ai Agent (see researcher_agent.py) instead of being served
over MCP/stdio.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Optional

from functools import lru_cache

from pydantic_ai import RunContext

from .. import config
from ..data_loader import get_review_by_id, get_reviews_by_place, load_places, load_reviews
from ..rag.index import get_index
from ..text_utils import district_matches


@dataclass
class ResearchDeps:
    """Per-run session state threaded through the researcher agent via
    RunContext. review_aliases lets search_reviews hand the model a short,
    easy-to-reproduce token (R1, R2, ...) instead of the long opaque
    reviewId strings in the dataset -- small local models reliably mangle
    the latter (dropped/swapped characters) when copying them into a
    citation, which silently fails grounding for a genuine citation. Kept
    alive across every researcher_agent run within one Coordinator.run()
    (including nudge re-runs) so alias numbering never collides with
    aliases already shown to the model in prior turns' message history.
    """

    review_aliases: dict[str, str] = field(default_factory=dict)  # alias -> real reviewId

    def alias_for(self, review_id: str) -> str:
        for alias, real_id in self.review_aliases.items():
            if real_id == review_id:
                return alias
        alias = f"R{len(self.review_aliases) + 1}"
        self.review_aliases[alias] = review_id
        return alias

    def resolve(self, alias: str) -> Optional[str]:
        return self.review_aliases.get(alias)

ASPECT_KEYWORDS = {
    "food": [
        "food",
        "dish",
        "meal",
        "taste",
        "flavor",
        "flavour",
        "delicious",
        "menu",
        "portion",
        "cooked",
        "meat",
        "fresh",
    ],
    "service": [
        "service",
        "staff",
        "waiter",
        "waitress",
        "friendly",
        "attentive",
        "rude",
        "slow",
        "wait",
        "server",
    ],
    "ambiance": [
        "atmosphere",
        "ambiance",
        "ambience",
        "decor",
        "cozy",
        "noisy",
        "view",
        "music",
        "seating",
        "outdoor",
    ],
    "occasion": [
        "lunch",
        "breakfast",
        "dinner",
        "brunch",
        "lunchtime",
    ],
    "price": [
        "price",
        "expensive",
        "cheap",
        "value",
        "worth",
        "pricey",
        "affordable",
        "cost",
    ],
}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OSRM_ROUTE_URL = "http://router.project-osrm.org/route/v1/driving"

GENERIC_CATEGORY_WORDS = {
    "restaurant",
    "restaurants",
    "food",
    "foods",
    "cuisine",
    "cuisines",
    "place",
    "places",
    "spot",
    "spots",
    "eatery",
    "eateries",
    "dining",
}


def _word_set(text: str) -> set[str]:
    return {w.strip(".,!?\"'").lower() for w in text.split() if len(w) > 2}


def _category_token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in GENERIC_CATEGORY_WORDS
    }


def _category_matches(query: str, categories: list[str] | None) -> bool:
    normalized_query = query.strip().lower()
    if not normalized_query:
        return False

    query_tokens = _category_token_set(normalized_query)

    for category in categories or []:
        normalized_category = category.strip().lower()
        if not normalized_category:
            continue
        if normalized_query in normalized_category or normalized_category in normalized_query:
            return True
        category_tokens = _category_token_set(normalized_category)
        if query_tokens and query_tokens <= category_tokens:
            return True

    return False


def list_places(
    categories: Optional[list[str] | str] = None,
    googleRating: Optional[float] = None,
    district: Optional[str] = None,
) -> list[dict]:
    """List every restaurant that exists in this dataset, with categories and Google rating.

    This is the authoritative, complete list of restaurants you have data for.
    If a cuisine or restaurant name you're looking for does not appear here
    (by name or category), it is NOT in the dataset -- do not assume it exists
    or recommend it from outside knowledge. The same applies to district: if
    filtering by district returns nothing, the dataset has no coverage there --
    do not recommend a restaurant from a different district as a substitute.

    Args:
        categories: optional list(array) of categories to filter by (e.g. ["Korean", "BBQ"]).
        googleRating: optional minimum Google rating to filter by (e.g. 4.0).
        district: optional Berlin district/neighborhood to filter by (e.g. "Mitte",
            "Neukölln", "Kreuzberg", "Charlottenburg-Wilmersdorf").
    """
    if isinstance(categories, str):
        categories = [categories]

    reviews = load_reviews()
    sampled_counts = collections.Counter(r.place_name for r in reviews)
    result = [
        {
            "placeName": name,
            "categories": place.categories,
            "googleRating": place.google_rating,
            "googleReviewCount": place.google_review_count,
            "address": place.address,
            "district": place.district,
            "sampledReviewCount": sampled_counts.get(name, 0),
        }
        for name, place in load_places().items()
    ]
    if categories:
        result = [
            r
            for r in result
            if any(_category_matches(c, r["categories"]) for c in categories)
        ]
    if googleRating:
        result = [r for r in result if r["googleRating"] and r["googleRating"] >= googleRating]
    if district:
        result = [r for r in result if district_matches(district, r["district"])]
    return result


def search_reviews(
    ctx: RunContext[ResearchDeps],
    query: str,
    place_name: Optional[str] = None,
    district: Optional[str] = None,
    top_k: int = 5,
) -> list[dict]:
    """Search review text for relevant snippets. This is your only source of
    evidence for restaurant-specific claims -- every reviewId returned here is
    the only kind of reviewId you're allowed to cite with [REVIEW:id].

    The reviewId returned here is a short local token (e.g. "R3"), not the
    dataset's real review ID -- copy it exactly as given, character for
    character.

    An empty or irrelevant result set means the dataset has no matching
    restaurant/cuisine/district for this query. Treat that as proof of
    absence, not as a reason to fall back on general knowledge -- call
    list_places to confirm before telling the user the dataset doesn't cover
    their request.

    Args:
        query: natural language search phrase (e.g. "good for vegetarians").
        place_name: optional exact/partial restaurant name to restrict search to.
        district: optional Berlin district/neighborhood to restrict search to
            (e.g. "Mitte", "Neukölln", "Kreuzberg", "Charlottenburg-Wilmersdorf").
            Use this whenever the user's request names a district/area.
        top_k: max number of results to return.
    """
    hits = get_index().search(query, place_name=place_name, district=district, top_k=top_k)
    places = load_places()
    return [
        {
            "reviewId": ctx.deps.alias_for(h.review.review_id),
            "placeName": h.review.place_name,
            "rating": h.review.rating,
            "googleRating": (places.get(h.review.place_name).google_rating if places.get(h.review.place_name) else None),
            "district": h.review.district,
            "text": h.review.text[:500],
            "language": h.review.language,
            "timestamp": h.review.timestamp,
            "relevanceScore": round(h.score, 4),
        }
        for h in hits
    ]


def get_place_stats(place_name: str) -> dict:
    """Return rating distribution and aspect-based sentiment signal for a restaurant.

    Aspects are computed deterministically from keyword frequency in positive
    (rating>=4) vs negative (rating<=2) reviews - no LLM involved.
    """
    reviews = get_reviews_by_place(place_name)
    if not reviews:
        return {"error": f"No reviews found for place matching '{place_name}'"}

    rating_dist = collections.Counter(r.rating for r in reviews if r.rating)
    avg_rating = sum(r.rating or 0 for r in reviews) / len(reviews)

    positive = [r for r in reviews if (r.rating or 0) >= 4]
    negative = [r for r in reviews if (r.rating or 0) <= 2]

    def aspect_counts(subset: list) -> dict[str, int]:
        counts = {aspect: 0 for aspect in ASPECT_KEYWORDS}
        for r in subset:
            words = _word_set(r.text)
            for aspect, keywords in ASPECT_KEYWORDS.items():
                if words & set(keywords):
                    counts[aspect] += 1
        return counts

    place = load_places().get(reviews[0].place_name)

    return {
        "placeName": reviews[0].place_name,
        "sampledReviewCount": len(reviews),
        "sampledAvgRating": round(avg_rating, 2),
        "googleRating": place.google_rating if place else None,
        "googleReviewCount": place.google_review_count if place else None,
        "ratingDistribution": dict(sorted(rating_dist.items())),
        "positiveReviewCount": len(positive),
        "negativeReviewCount": len(negative),
        "aspectMentionsInPositiveReviews": aspect_counts(positive),
        "aspectMentionsInNegativeReviews": aspect_counts(negative),
    }


@lru_cache(maxsize=1)
def _get_nli_pipeline():
    from transformers import pipeline

    return pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")


def split_sentences(text: str) -> list[str]:
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]


def verify_quote(review_id: str, claim: str) -> dict:
    review = get_review_by_id(review_id)
    nli = _get_nli_pipeline()

    sentences = split_sentences(review.text)

    # Check single sentences, plus 2-sentence windows. A claim that
    # synthesizes two facts from adjacent review sentences ("affordable and
    # family-friendly") is genuinely grounded but no single sentence entails
    # the whole claim on its own -- only checking single sentences under-
    # scores (and rejects) real citations.
    premises = list(sentences)
    premises.extend(
        f"{sentences[i]} {sentences[i + 1]}" for i in range(len(sentences) - 1)
    )

    best_evidence = None
    best_score = 0.0

    for premise in premises:
        result = nli(
            f"{premise} [SEP] {claim}"
        )[0]

        label = result["label"].upper()
        score = float(result["score"])

        if label == "ENTAILMENT" and score > best_score:
            best_score = score
            best_evidence = premise

    grounded = best_evidence is not None and best_score >= config.MIN_ENTAILMENT_CONFIDENCE

    return {
        "review_id": review_id,
        "claim": claim,
        "grounded": grounded,
        "confidence": best_score,
        "evidence": best_evidence,
    }

def _find_place(place_name: str):
    for name, place in load_places().items():
        if place_name.lower() in name.lower():
            return place
    return None


def get_place_address(place_name: str) -> dict:
    """Return the known street address for a restaurant matching the given
    (possibly partial) name from the dataset.
    """
    place = _find_place(place_name)
    if place is None:
        return {"error": f"No place found matching '{place_name}'"}
    return {
        "placeName": place.name,
        "placeAddress": place.address,
    }