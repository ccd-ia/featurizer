"""Unit tests for the text Path-1 φ-bridges (no database).

Hand-computed expectations on tiny documents: sentiment valence means,
Fernández-Huerta / Flesch readability, stopword-profile language id, and the
NER multi-column mapping via a fake spaCy pipeline (the real model is a
separate download; a skip-gated smoke covers it when present). A Spanish +
English pair proves the multilingual default in each family.
"""

from __future__ import annotations

from datetime import date

import pytest

from featurizer.bridge import (
    LanguageIdBridge,
    NERCountsBridge,
    ReadabilityBridge,
    SentimentBridge,
)
from featurizer.bridge.nlp import _syllables_en, _syllables_es

# --------------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------------- #


def test_sentiment_spanish_default_hand_values():
    bridge = SentimentBridge(pk_col="id", text_col="body")
    assert bridge.language == "es"
    phi = bridge.compute(
        [
            {"id": 1, "body": "Excelente servicio, muy bueno."},
            {"id": 2, "body": "Terrible retraso otra vez"},
            {"id": 3, "body": "xyzzy plugh"},  # no lexicon evidence -> NULL
        ],
        fit_rows=[],
    )
    assert phi[1] == pytest.approx((0.9 + 0.6) / 2)  # excelente + bueno
    assert phi[2] == pytest.approx((-0.9 + -0.6) / 2)  # terrible + retraso
    assert phi[3] is None


def test_sentiment_english_is_a_one_line_override():
    bridge = SentimentBridge(pk_col="id", text_col="body", language="en")
    phi = bridge.compute([{"id": 1, "body": "great work, thanks"}], fit_rows=[])
    assert phi[1] == pytest.approx((0.8 + 0.6) / 2)  # great + thanks


def test_sentiment_multilingual_lexicon_scores_both_languages():
    bridge = SentimentBridge(pk_col="id", text_col="body", language="xx")
    phi = bridge.compute(
        [
            {"id": "es", "body": "excelente"},
            {"id": "en", "body": "terrible delay"},
        ],
        fit_rows=[],
    )
    assert phi["es"] == pytest.approx(0.9)
    assert phi["en"] == pytest.approx((-0.9 + -0.6) / 2)


def test_sentiment_custom_lexicon_and_unknown_language():
    bridge = SentimentBridge(
        pk_col="id", text_col="body", language="nah", lexicon={"tlazohtla": 0.9}
    )
    assert bridge.compute([{"id": 1, "body": "tlazohtla"}], fit_rows=[])[1] == 0.9
    with pytest.raises(ValueError, match="no built-in lexicon"):
        SentimentBridge(pk_col="id", text_col="body", language="nah")


# --------------------------------------------------------------------------- #
# Readability
# --------------------------------------------------------------------------- #


def test_spanish_syllable_heuristic():
    assert _syllables_es("perro") == 2
    assert _syllables_es("poeta") == 3  # strong-strong hiatus o-e
    assert _syllables_es("día") == 2  # accented-weak hiatus í-a
    assert _syllables_es("país") == 2
    assert _syllables_es("bien") == 1  # diphthong


def test_english_syllable_heuristic():
    assert _syllables_en("cat") == 1
    assert _syllables_en("table") == 2  # -le keeps its syllable
    assert _syllables_en("make") == 1  # silent trailing e
    assert _syllables_en("beautiful") == 3


def test_readability_fernandez_huerta_hand_value():
    bridge = ReadabilityBridge(pk_col="id", text_col="body")  # es default
    phi = bridge.compute([{"id": 1, "body": "El perro come."}], fit_rows=[])
    # 5 syllables / 3 words / 1 sentence:
    # 206.84 - 0.60 * (5/3*100) - 1.02 * 3 = 103.78
    assert phi[1] == pytest.approx(103.78)


def test_readability_flesch_hand_value():
    bridge = ReadabilityBridge(pk_col="id", text_col="body", language="en")
    phi = bridge.compute([{"id": 1, "body": "The cat sat."}], fit_rows=[])
    # 3 syllables / 3 words / 1 sentence:
    # 206.835 - 1.015 * 3 - 84.6 * 1 = 119.19
    assert phi[1] == pytest.approx(119.19)


def test_readability_empty_document_is_null_and_language_validated():
    bridge = ReadabilityBridge(pk_col="id", text_col="body")
    assert bridge.compute([{"id": 1, "body": None}], fit_rows=[])[1] is None
    with pytest.raises(ValueError, match="language must be"):
        ReadabilityBridge(pk_col="id", text_col="body", language="fr")


# --------------------------------------------------------------------------- #
# Language id
# --------------------------------------------------------------------------- #


def test_language_id_detects_es_en_pt():
    bridge = LanguageIdBridge(pk_col="id", text_col="body")
    phi = bridge.compute(
        [
            {"id": 1, "body": "El perro come en la casa y no quiere salir"},
            {"id": 2, "body": "The dog eats in the house and does not leave"},
            {"id": 3, "body": "O cachorro come na casa e não quer sair"},
            {"id": 4, "body": "xyzzy 42"},  # no evidence -> NULL
        ],
        fit_rows=[],
    )
    assert phi[1] == {"language": "es"}
    assert phi[2] == {"language": "en"}
    assert phi[3] == {"language": "pt"}
    assert phi[4] == {"language": None}


def test_language_id_is_categorical_in_emit_yaml():
    bridge = LanguageIdBridge(pk_col="id", text_col="body")
    fragment = bridge.emit_yaml(
        output_table="t", pk="id", parent_alias="p", parent_key="id", fk="id"
    )
    assert fragment["entity"]["variables"] == {"language": {"type": "categorical"}}


def test_language_id_candidate_set_validated():
    with pytest.raises(ValueError, match="no built-in profile"):
        LanguageIdBridge(pk_col="id", text_col="body", languages=["es", "klingon"])


# --------------------------------------------------------------------------- #
# NER counts (fake pipeline: our mapping logic, not spaCy, is under test)
# --------------------------------------------------------------------------- #


class _Ent:
    def __init__(self, label: str) -> None:
        self.label_ = label


class _Doc:
    def __init__(self, *labels: str) -> None:
        self.ents = [_Ent(label) for label in labels]


class _FakeNLP:
    def __init__(self, docs) -> None:
        self.docs = docs

    def pipe(self, texts):
        assert len(list(texts)) == len(self.docs)
        return self.docs


def test_ner_counts_multicolumn_mapping(monkeypatch):
    bridge = NERCountsBridge(pk_col="id", text_col="body")
    monkeypatch.setattr(
        bridge,
        "_load_nlp",
        lambda: _FakeNLP(
            [
                _Doc("PER", "ORG", "LOC"),  # es/xx-style labels
                _Doc("PERSON", "GPE", "MONEY", "DATE", "PERSON"),  # en-style
                _Doc(),
            ]
        ),
    )
    phi = bridge.compute(
        [{"id": 1, "body": "a"}, {"id": 2, "body": "b"}, {"id": 3, "body": "c"}],
        fit_rows=[],
    )
    assert phi[1] == {
        "persons": 1.0,
        "orgs": 1.0,
        "locations": 1.0,
        "money": 0.0,
        "dates": 0.0,
    }
    assert phi[2] == {
        "persons": 2.0,
        "orgs": 0.0,
        "locations": 1.0,
        "money": 1.0,
        "dates": 1.0,
    }
    assert phi[3] == {
        "persons": 0.0,
        "orgs": 0.0,
        "locations": 0.0,
        "money": 0.0,
        "dates": 0.0,
    }


def test_ner_model_selection_and_vintage():
    assert NERCountsBridge(pk_col="i", text_col="t").model == "xx_ent_wiki_sm"
    assert (
        NERCountsBridge(pk_col="i", text_col="t", language="es").model
        == "es_core_news_sm"
    )
    custom = NERCountsBridge(
        pk_col="i",
        text_col="t",
        model="es_core_news_lg",
        model_vintage=date(2023, 4, 1),
    )
    assert custom.model == "es_core_news_lg"
    assert custom.metadata["model_vintage"] == date(2023, 4, 1)
    custom.assert_model_vintage(date(2024, 1, 1))  # vintage predates: fine
    with pytest.raises(ValueError, match="not knowable"):
        custom.assert_model_vintage(date(2022, 1, 1))


def test_ner_real_model_smoke_when_available():
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("es_core_news_sm")
    except OSError:
        pytest.skip("es_core_news_sm not downloaded")
    bridge = NERCountsBridge(pk_col="id", text_col="body", language="es")
    phi = bridge.compute(
        [{"id": 1, "body": "Juan Pérez trabaja en Petróleos Mexicanos."}],
        fit_rows=[],
    )
    assert phi[1]["persons"] >= 1.0
    assert phi[1]["orgs"] >= 1.0
