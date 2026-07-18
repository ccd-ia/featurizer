# coding: utf-8

"""Text φ-bridges, Path 1 of the text substrate: reduce → aggregate.

Each bridge collapses a document column to per-event scalars; the SQL spine
then trends them (sentiment slope, recency of last negative event, volatility)
with its normal causal bound. Per the taxonomy's multilingual rule, defaults
are Spanish/multilingual — English is a one-line ``language=`` override, never
the silent default.

Three of the four families are **dependency-free by design** — a curated
valence lexicon, syllable-formula readability, stopword-profile language id —
so they run in the plain ``uv sync`` environment. Only
:class:`NERCountsBridge` needs spaCy (the ``[bridge]`` extra) and a downloaded
model; it is also the one *pretrained*-model bridge here, so it carries the
``model_vintage`` caveat (ADR-0014): ``assert_pre_t0`` cannot see a model
trained on post-t₀ text — declare the vintage and assert it in strict
backtests.

None of these bridges fits anything (``fit_rows`` is unused): φ reads only the
row's own content, so temporal correctness is entirely the spine's aggregation
bound.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .base import BridgeComputer, MultiColumnBridge

# --------------------------------------------------------------------------- #
# Shared tokenization
# --------------------------------------------------------------------------- #


def word_tokens(text: str) -> List[str]:
    """Lowercased word tokens, Unicode-aware (keeps áéíóúüñ etc.)."""
    return re.findall(r"[^\W\d_]+", text.lower())


def _sentences(text: str) -> int:
    """Sentence count (>= 1 for non-empty text)."""
    return max(1, len([s for s in re.split(r"[.!?…]+", text) if s.strip()]))


# --------------------------------------------------------------------------- #
# Sentiment: dictionary valence scores, Spanish-register default
# --------------------------------------------------------------------------- #


def _lex(*groups: Tuple[float, str]) -> Dict[str, float]:
    return {word: value for value, words in groups for word in words.split()}


#: Compact starter lexicons (word -> valence in [-1, 1]). Deliberately small
#: and legible: production deployments should pass their own domain lexicon
#: (legal, logistics, electoral register) via ``lexicon=`` — the reference
#: lexicons here exist so the multilingual default works out of the box and
#: the tests are exact. NRC/LIWC-style licensed lexicons are *not* vendored.
SENTIMENT_LEXICONS: Dict[str, Dict[str, float]] = {
    "es": _lex(
        (0.9, "excelente maravilloso maravillosa fantástico fantástica perfecto"),
        (0.9, "perfecta éxito exitoso hermoso hermosa"),
        (
            0.6,
            "bueno buena bien mejor amable feliz alegre agradable positivo "
            "positiva satisfecho satisfecha seguro segura tranquilo tranquila "
            "limpio limpia correcto correcta cumple aprobado aprobada favorable "
            "eficiente rápido rápida confiable gracias resuelto resuelta",
        ),
        (
            -0.6,
            "malo mala peor triste sucio sucia problema problemas queja quejas "
            "falla fallas error errores retraso retrasos rechazado rechazada "
            "deficiente negativo negativa daño dañado dañada riesgo grave "
            "peligro peligroso peligrosa incumplimiento crítico crítica",
        ),
        (-0.9, "terrible horrible pésimo pésima violación violaciones"),
    ),
    "en": _lex(
        (0.8, "excellent wonderful fantastic perfect great success successful"),
        (
            0.6,
            "good better best happy pleasant beautiful positive satisfied safe "
            "clean correct approved favorable efficient fast reliable thanks "
            "nice improved resolved timely compliant love",
        ),
        (
            -0.6,
            "bad worse sad angry dirty problem problems complaint complaints "
            "failure failures error errors delay delays rejected deficient "
            "negative damage damaged risk poor broken unsafe late severe "
            "dangerous critical",
        ),
        (-0.8, "terrible horrible awful worst breach violation violations"),
    ),
}
SENTIMENT_LEXICONS["xx"] = {**SENTIMENT_LEXICONS["en"], **SENTIMENT_LEXICONS["es"]}


class SentimentBridge(BridgeComputer):
    """Lexicon-based valence per document (dependency-free).

    φ = mean valence of the document's lexicon-matched tokens, in [-1, 1];
    NULL when no token matches (no evidence ≠ neutral). The default lexicon is
    Spanish-register (``language="es"``); ``"en"`` and ``"xx"`` (both) are
    built in, and ``lexicon=`` accepts a custom ``{word: valence}`` mapping —
    the intended path for licensed or domain lexicons. Downstream, the spine
    provides sentiment slope / recency-of-negative / volatility for free.
    """

    def __init__(
        self,
        *,
        pk_col: str,
        text_col: str,
        language: str = "es",
        lexicon: Optional[Dict[str, float]] = None,
        name: str = "sentiment",
        value_col: str = "sentiment",
    ) -> None:
        super().__init__(name=name, value_col=value_col, value_type="numeric")
        self.pk_col = pk_col
        self.text_col = text_col
        self.language = language
        if lexicon is None:
            if language not in SENTIMENT_LEXICONS:
                raise ValueError(
                    f"{name}: no built-in lexicon for language={language!r} "
                    f"(built-ins: {sorted(SENTIMENT_LEXICONS)}); pass lexicon= "
                    "with your own {word: valence} mapping."
                )
            lexicon = SENTIMENT_LEXICONS[language]
        self.lexicon = lexicon

    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, Optional[float]]:
        out: Dict[Any, Optional[float]] = {}
        for row in rows:
            scores = [
                self.lexicon[w]
                for w in word_tokens(str(row.get(self.text_col) or ""))
                if w in self.lexicon
            ]
            out[row[self.pk_col]] = sum(scores) / len(scores) if scores else None
        return out


# --------------------------------------------------------------------------- #
# Readability: syllable-formula scores (Fernández-Huerta / Flesch)
# --------------------------------------------------------------------------- #

_ES_VOWELS = "aeiouáéíóúü"
_ES_STRONG = "aeoáéó"


def _syllables_es(word: str) -> int:
    """Spanish syllable count: vowel groups, splitting strong-strong hiatus
    and accented-weak (í/ú) hiatus. A documented approximation."""
    count = 0
    prev_vowel = False
    prev_strong = False
    prev_accented = False
    for ch in word:
        is_vowel = ch in _ES_VOWELS
        if is_vowel:
            if not prev_vowel:
                count += 1
            elif ch in "íú":
                count += 1
            elif ch in _ES_STRONG and (prev_strong or prev_accented):
                count += 1
        prev_vowel = is_vowel
        prev_strong = is_vowel and ch in _ES_STRONG
        prev_accented = is_vowel and ch in "íú"
    return max(count, 1)


def _syllables_en(word: str) -> int:
    """English syllable count: vowel groups minus a silent trailing 'e'.
    The classic heuristic; documented approximation."""
    groups = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith("le") and groups > 1:
        groups -= 1
    return max(groups, 1)


class ReadabilityBridge(BridgeComputer):
    """Readability score per document (dependency-free).

    ``language="es"`` (default) uses the Fernández-Huerta index — the standard
    Spanish adaptation of Flesch — ``206.84 - 0.60·P - 1.02·F`` (P = syllables
    per 100 words, F = words per sentence). ``language="en"`` uses Flesch
    Reading Ease. Higher = easier. NULL for empty documents. Syllable counting
    is heuristic (see the helpers); scores are comparable within a corpus,
    which is all the spine's trends need.
    """

    def __init__(
        self,
        *,
        pk_col: str,
        text_col: str,
        language: str = "es",
        name: str = "readability",
        value_col: str = "readability",
    ) -> None:
        super().__init__(name=name, value_col=value_col, value_type="numeric")
        if language not in ("es", "en"):
            raise ValueError(f"{name}: language must be 'es' or 'en', got {language!r}")
        self.pk_col = pk_col
        self.text_col = text_col
        self.language = language

    def _score(self, text: str) -> Optional[float]:
        words = word_tokens(text)
        if not words:
            return None
        sentences = _sentences(text)
        if self.language == "es":
            syllables = sum(_syllables_es(w) for w in words)
            per_100 = 100.0 * syllables / len(words)
            return 206.84 - 0.60 * per_100 - 1.02 * (len(words) / sentences)
        syllables = sum(_syllables_en(w) for w in words)
        return (
            206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
        )

    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, Optional[float]]:
        return {
            row[self.pk_col]: self._score(str(row.get(self.text_col) or ""))
            for row in rows
        }


# --------------------------------------------------------------------------- #
# Language id: stopword-profile detection (dependency-free)
# --------------------------------------------------------------------------- #

#: Equal-sized profiles of high-frequency function words per language. The
#: argmax of profile hits is the detected language; ties resolve in this
#: declaration order (es first, per the multilingual-default rule).
LANGUAGE_PROFILES: Dict[str, frozenset[str]] = {
    "es": frozenset(
        "el la los las de del que y en un una es al lo como más pero sus le "
        "ya o porque muy sin sobre también hasta donde desde nos".split()
    ),
    "en": frozenset(
        "the of and to in is that it for on was with as be at by this not "
        "are but from or have an they which you were all there when".split()
    ),
    "pt": frozenset(
        "o a os as de do da que e em um uma não com por mais para como mas "
        "foi ao ele das tem à seu sua ou ser está".split()
    ),
    "fr": frozenset(
        "le la les de des et en un du une que est pour qui dans par plus pas "
        "au sur ne se ce il sont avec son cette mais où".split()
    ),
}


class LanguageIdBridge(MultiColumnBridge):
    """Detected language code per document (dependency-free).

    Counts hits against equal-sized stopword profiles and returns the argmax
    code as a **categorical** column (the ADR-0007 one-hot path downstream);
    NULL when no profile word occurs (no evidence). ``languages=`` restricts
    the candidate set. Deterministic — ties resolve in profile declaration
    order, Spanish first.
    """

    def __init__(
        self,
        *,
        pk_col: str,
        text_col: str,
        languages: Sequence[str] = ("es", "en", "pt", "fr"),
        name: str = "language_id",
        value_col: str = "language",
    ) -> None:
        unknown = [lang for lang in languages if lang not in LANGUAGE_PROFILES]
        if unknown:
            raise ValueError(
                f"{name}: no built-in profile for {unknown} "
                f"(built-ins: {sorted(LANGUAGE_PROFILES)})"
            )
        super().__init__(
            name=name,
            value_cols=[value_col],
            value_types={value_col: "categorical"},
        )
        self.pk_col = pk_col
        self.text_col = text_col
        self.languages = list(languages)
        self.value_col = value_col

    def _detect(self, text: str) -> Optional[str]:
        words = word_tokens(text)
        best: Optional[str] = None
        best_hits = 0
        for lang in self.languages:
            hits = sum(w in LANGUAGE_PROFILES[lang] for w in words)
            if hits > best_hits:
                best, best_hits = lang, hits
        return best

    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, Dict[str, Any]]:
        return {
            row[self.pk_col]: {
                self.value_col: self._detect(str(row.get(self.text_col) or ""))
            }
            for row in rows
        }


# --------------------------------------------------------------------------- #
# NER counts: spaCy, multilingual default, multi-column (one parse, N columns)
# --------------------------------------------------------------------------- #

#: spaCy entity label -> output column. Models lacking a type (the multilingual
#: and Spanish models tag no MONEY/DATE) simply count 0 there — columns are
#: stable across models so downstream configs never change shape.
NER_LABEL_COLUMNS: Dict[str, str] = {
    "PER": "persons",
    "PERSON": "persons",
    "ORG": "orgs",
    "LOC": "locations",
    "GPE": "locations",
    "MONEY": "money",
    "DATE": "dates",
}

#: ``language=`` -> default spaCy model (override with ``model=``).
NER_DEFAULT_MODELS: Dict[str, str] = {
    "xx": "xx_ent_wiki_sm",
    "es": "es_core_news_sm",
    "en": "en_core_web_sm",
}


class NERCountsBridge(MultiColumnBridge):
    """Named-entity counts per document — one spaCy parse, five columns.

    ``compute()`` returns ``{pk: {persons, orgs, locations, money, dates}}``
    (the ADR-0014 multi-column contract). The default model is multilingual
    (``language="xx"`` → ``xx_ent_wiki_sm``); ``"es"``/``"en"`` map to their
    ``*_core_news/web_sm`` models and ``model=`` overrides entirely. Models
    are separate downloads: ``python -m spacy download <model>``.

    This is a *pretrained* model: ``assert_pre_t0`` cannot guard it. Declare
    ``model_vintage=`` (the model's training-cutoff date) and call
    :meth:`assert_model_vintage` in strict backtests (ADR-0014).
    """

    VALUE_COLS = ("persons", "orgs", "locations", "money", "dates")

    def __init__(
        self,
        *,
        pk_col: str,
        text_col: str,
        language: str = "xx",
        model: Optional[str] = None,
        name: str = "ner_counts",
        model_vintage: Any = None,
    ) -> None:
        super().__init__(
            name=name, value_cols=list(self.VALUE_COLS), model_vintage=model_vintage
        )
        self.pk_col = pk_col
        self.text_col = text_col
        self.language = language
        self.model = model or NER_DEFAULT_MODELS.get(language, NER_DEFAULT_MODELS["xx"])

    def _load_nlp(self) -> Any:
        try:
            import spacy  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "NERCountsBridge needs spaCy: "
                "install with `pip install 'featurizer[bridge]'`."
            ) from exc
        try:
            return spacy.load(self.model)
        except OSError as exc:
            raise OSError(
                f"NERCountsBridge: spaCy model {self.model!r} is not installed. "
                f"Download it with `python -m spacy download {self.model}`."
            ) from exc

    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, Dict[str, Any]]:
        nlp = self._load_nlp()
        texts: Iterable[str] = [str(r.get(self.text_col) or "") for r in rows]
        out: Dict[Any, Dict[str, Any]] = {}
        for row, doc in zip(rows, nlp.pipe(texts)):
            counts: Dict[str, Any] = {col: 0.0 for col in self.value_cols}
            for ent in doc.ents:
                col = NER_LABEL_COLUMNS.get(ent.label_)
                if col is not None:
                    counts[col] += 1.0
            out[row[self.pk_col]] = counts
        return out
