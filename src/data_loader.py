"""Loads the scraped Google Maps dataset and builds place-level aggregates."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from . import config


@dataclass
class Place:
    name: str
    address: Optional[str]
    place_id: Optional[str]
    categories: list[str]
    google_rating: Optional[float]
    google_review_count: Optional[int]
    phone: Optional[str]
    website: Optional[str]
    description: Optional[str]
    district: Optional[str] = None


@dataclass
class Review:
    review_id: str
    place_name: str
    rating: Optional[int]
    text: str
    language: Optional[str]
    timestamp: Optional[str]
    owner_response: Optional[str] = None
    district: Optional[str] = None


def _district_from_filename(path: Path) -> str:
    """Derive a human-readable district name from a districtwise dataset
    filename, e.g. 'dataset_google-maps-reviews-scraper_neukolln.json' ->
    'Neukölln'. Falls back to the raw stem for filenames that don't match
    the expected scraper naming convention."""
    stem = path.stem
    prefix = "dataset_google-maps-reviews-scraper_"
    raw = stem[len(prefix):] if stem.startswith(prefix) else stem
    # Known scraper filenames use plain ASCII for umlauts (neukolln).
    display = raw.replace("-", "-").replace("_", " ")
    fixes = {"neukolln": "Neukölln"}
    return fixes.get(display.lower(), display.title() if display.islower() else display)


@lru_cache(maxsize=1)
def load_raw() -> list[dict]:
    """Load and merge every *.json dataset file in config.DATA_DIR.

    Rows are deduped by reviewId so the same review scraped into multiple
    exports (e.g. restaurants re-scraped across runs) is only counted once.
    Rows whose address doesn't look like Berlin, Germany are dropped -- the
    scraper occasionally picks up same-named places in other "Berlin"s (e.g.
    Berlin, Maryland, US) since it matches on place name, not geography.
    """
    seen_ids: set[str] = set()
    merged: list[dict] = []
    json_files = sorted(config.DATA_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No dataset JSON files found in {config.DATA_DIR}")

    for path in json_files:
        district = _district_from_filename(path)
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            address = (row.get("placeAddress") or "").lower()
            if address and "germany" not in address and "deutschland" not in address:
                continue
            review_id = row.get("reviewId")
            if review_id is not None and review_id in seen_ids:
                continue
            if review_id is not None:
                seen_ids.add(review_id)
            row["_district"] = district
            merged.append(row)
    return merged


@lru_cache(maxsize=1)
def load_reviews() -> list[Review]:
    reviews = []
    for row in load_raw():
        text = (row.get("text") or "").strip()
        if not text:
            continue
        reviews.append(
            Review(
                review_id=str(row.get("reviewId")),
                place_name=row["placeName"],
                rating=row.get("rating"),
                text=text,
                language=row.get("language"),
                timestamp=row.get("timestamp"),
                owner_response=(row.get("ownerResponseText") or "").strip() or None,
                district=row.get("_district"),
            )
        )
    return reviews


@lru_cache(maxsize=1)
def load_places() -> dict[str, Place]:
    places: dict[str, Place] = {}
    for row in load_raw():
        name = row["placeName"]
        if name in places:
            continue
        places[name] = Place(
            name=name,
            address=row.get("placeAddress"),
            place_id=row.get("placePlaceId"),
            categories=list(row.get("placeCategories") or []),
            google_rating=row.get("placeRating"),
            google_review_count=row.get("placeReviewCount"),
            phone=row.get("placePhone"),
            website=row.get("placeWebsite"),
            description=row.get("placeDescription"),
            district=row.get("_district"),
        )
    return places


@lru_cache(maxsize=1)
def _reviews_by_id() -> dict[str, Review]:
    return {r.review_id: r for r in load_reviews()}


@lru_cache(maxsize=1)
def _reviews_by_place() -> dict[str, list[Review]]:
    by_place: dict[str, list[Review]] = {}
    for r in load_reviews():
        by_place.setdefault(r.place_name, []).append(r)
    return by_place


def get_review_by_id(review_id: str) -> Optional[Review]:
    return _reviews_by_id().get(str(review_id))


def get_reviews_by_place(place_name: str) -> list[Review]:
    """Reviews for restaurants whose name contains `place_name` (case-insensitive)."""
    needle = place_name.lower()
    matches: list[Review] = []
    for name, reviews in _reviews_by_place().items():
        if needle in name.lower():
            matches.extend(reviews)
    return matches
