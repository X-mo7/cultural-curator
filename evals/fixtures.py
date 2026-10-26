"""Synthetic judgments and controlled vectors; no personal data or live models.

Relevance is a direct annotation of the text and persona, never a threshold on
an embedding or ranking score. Vector fixtures below deliberately include both
retrieval errors and a lexical distractor. These are regression scenarios,
not estimates of Mistral's real-world quality.
"""

from datetime import datetime, timezone

from cultural_curator.domain import CulturalEvent, CulturalMemory, Preference
from cultural_curator.ranking.semantic import event_text
from cultural_curator.retrieval import FixtureEmbedder


MEMORIES = tuple(
    CulturalMemory(id=identifier, text=text, created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    for identifier, text in (
        ("gallery", "I enjoyed a calm afternoon inside an art gallery."),
        ("photo", "I loved the street photography exhibition in Paris."),
        ("painting", "I admired vivid paintings of people and landscapes."),
        ("theatre", "I attended Moliere's play and enjoyed the actors."),
        ("poster", "I noticed a stage performance poster, but did not attend."),
    )
)

# Queries concern experienced cultural outings. A poster is not attendance.
# The phrase "stage performance" is a synonym of the attended play, while the
# poster is an intentional lexical distractor. Both errors are visible in output.
MEMORY_QUERIES = {
    "photo": "photography Paris",
    "painting": "vivid paintings",
    "theatre": "stage performance",
}
MEMORY_QRELS = {
    "photo": frozenset({"photo"}),
    "painting": frozenset({"painting"}),
    "theatre": frozenset({"theatre"}),
}

EVENTS = (
    CulturalEvent(id="photo", title="Street portraits", description="Urban photography."),
    CulturalEvent(id="painting", title="Faces in colour", description="Figurative paintings."),
    CulturalEvent(id="theatre", title="Moliere on stage", description="A classical play."),
    CulturalEvent(id="sound", title="Electronic nights", description="Loud experimental sound."),
)
PERSONAS = {
    "visual": (
        Preference(id="taste-photo", label="street photography", polarity="positive", origin="explicit"),
        Preference(id="avoid-sound", label="loud sound", polarity="negative", origin="explicit"),
    ),
    "theatre": (
        Preference(id="taste-theatre", label="classical theatre", polarity="positive", origin="explicit"),
    ),
}
# Fixed text-based grades: 3=strong match, 1=plausible secondary interest,
# 0=irrelevant or conflicting. The visual persona values documentary imagery;
# the theatre persona wants a play, not merely another live performance.
EVENT_QRELS = {
    "visual": {"photo": 3, "painting": 1, "theatre": 0, "sound": 0},
    "theatre": {"photo": 0, "painting": 0, "theatre": 3, "sound": 0},
}


def memory_embedder() -> FixtureEmbedder:
    """Controlled semantic scores, separate from the judgments above."""
    memory_vectors = {
        "gallery": [0.95, 0.0, 0.1, 0.0],
        "photo": [0.8, 0.6, 0.0, 0.0],
        "painting": [0.0, 0.9, 0.1, 0.0],
        "theatre": [0.0, 0.0, 1.0, 0.0],
        "poster": [0.0, 0.0, 0.9, 0.1],
    }
    return FixtureEmbedder({
        **{memory.text: memory_vectors[memory.id] for memory in MEMORIES},
        MEMORY_QUERIES["photo"]: [1.0, 0.0, 0.0, 0.0],
        MEMORY_QUERIES["painting"]: [0.0, 1.0, 0.0, 0.0],
        MEMORY_QUERIES["theatre"]: [0.0, 0.0, 1.0, 0.0],
        "astronomy": [0.0, 0.0, 0.0, 1.0],
    })


def event_embedder() -> FixtureEmbedder:
    event_vectors = {
        "photo": [1.0, 0.0, 0.0],
        "painting": [0.8, 0.6, 0.0],
        "theatre": [0.0, 1.0, 0.0],
        "sound": [0.0, 0.3, 0.7],
    }
    return FixtureEmbedder({
        **{event_text(event): event_vectors[event.id] for event in EVENTS},
        "street photography": [1.0, 0.0, 0.0],
        "loud sound": [0.0, 0.0, 1.0],
        "classical theatre": [0.0, 1.0, 0.0],
    })
