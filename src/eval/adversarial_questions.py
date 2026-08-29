"""Adversarial + normal test questions for the grounding eval harness.

`should_refuse=True` marks questions about cuisines/requests that do NOT
exist in the dataset -- the agent must decline rather than hallucinate a
recommendation (see the ramen example this project is built around).

Ground truth here must match the current data/*.json files -- verify with
`search_reviews`/`list_places` before flipping `should_refuse`, since the
dataset's coverage changes whenever files are added under data/.
"""
from dataclasses import dataclass


@dataclass
class EvalQuestion:
    query: str
    should_refuse: bool
    note: str = ""


QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        "Where can I get good ramen?",
        should_refuse=False,
        note="Niko Niko Ramen and Morimori Ramen (both Kreuzberg) are ramen restaurants "
        "with 1300+/2800+ Google reviews each. Was should_refuse=True; flipped after "
        "verifying against current data -- see module docstring on stale ground truth.",
    ),
    EvalQuestion(
        "Is there a Peruvian restaurant in Berlin?",
        should_refuse=True,
        note="No Peruvian restaurants anywhere in the dataset (list_places(categories=['Peruvian']) is empty).",
    ),
    EvalQuestion(
        "Is there a Nepalese sushi restaurant?",
        should_refuse=True,
        note="Nepalese (Bajra Nepalesisches Restaurant, My Tibet Haus) and Sushi restaurants "
        "both exist individually, but no restaurant is tagged both -- genuine partial-match refusal.",
    ),
    EvalQuestion(
        "Are there any good bars in Mitte?",
        should_refuse=False,
        note="Pantry, LAWRENCE berlin mitte, and Cantina Mexicana Que Pasa are bars/cocktail "
        "bars in the Mitte district file.",
    ),
    EvalQuestion(
        "Where can I find Ethiopian food in Charlottenburg?",
        should_refuse=True,
        note="Ethiopian restaurants exist in the dataset (Lalibela x3, in Kreuzberg/Neukölln) "
        "but none in Charlottenburg -- district-scoped trap, distinct from a cuisine-absent refusal.",
    ),
    EvalQuestion(
        "What are good coffee shops with breakfast in Neukölln?",
        should_refuse=False,
        note="Ubercoffee & Bakery Neukölln and MaraLou Café are real cafes in the Neukölln district file.",
    ),
    EvalQuestion(
        "Is there a good Korean BBQ place?",
        should_refuse=False,
        note="Kimchi Princess and BBQ Kitchen are Korean BBQ.",
    ),
    EvalQuestion(
        "Where's good for Middle Eastern or halal food?",
        should_refuse=False,
        note="Mann-o-Salwa is halal/Middle Eastern.",
    ),
    EvalQuestion(
        "Persian food in Berlin?",
        should_refuse=False,
        note="Karun Bistro is categorized as Persian restaurant.",
    ),
    EvalQuestion(
        "Is there an Indian restaurant?",
        should_refuse=False,
        note="AMRIT and Saravanaa Bhavan are Indian.",
    ),
    EvalQuestion(
        "What are good lunch options?",
        should_refuse=False,
        note="Restaurant Facil, BLESS, Hofbräu have lunch mentions.",
    ),
    EvalQuestion(
        "Is there a good Italian restaurant?",
        should_refuse=False,
        note="Mio Berlin is Italian.",
    ),
    EvalQuestion(
        "Where can I get Israeli food?",
        should_refuse=False,
        note="NENI Berlin is Israeli.",
    ),
    EvalQuestion(
        "What is the best place for sushi in Berlin?",
        should_refuse=False,
        note="893 Ryotei Japanese Restaurant has multiple reviews explicitly praising its sushi.",
    ),
        EvalQuestion(
        "What are top 3 Italian restaurants in Berlin?",
        should_refuse=False,
        note="Oliveto, Babbo bar are Italian.",
    ),
]
