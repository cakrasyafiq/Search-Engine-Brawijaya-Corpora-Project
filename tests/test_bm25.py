import pytest
from app.search.tokenizer import tokenize, normalize_text, strip_accents
from app.search.bm25_index import BM25InvertedIndex, DocumentRecord


def test_tokenizer_accents_and_compounds():
    # Accented text
    assert strip_accents("Borèh") == "Boreh"
    assert strip_accents("pètro") == "petro"

    # Compound words
    tokens = tokenize("Among-among rite")
    assert "among-among" in tokens
    assert "among" in tokens
    assert "rite" in tokens

    # Normalization
    assert normalize_text("  BORÈH  ") == "boreh"


def test_bm25_ranking_accuracy():
    index = BM25InvertedIndex(k1=1.5, b=0.75)

    doc1 = DocumentRecord(
        id=1,
        term="Among-among",
        culture="Tengger",
        semantic_domain="Ritual",
        definition="An Among-among is Tengger's protection rite for 44-day-old babies.",
        variants="selapanan\nslametan bayi\nmitoni cilik\nbancakan anak",
    )

    doc2 = DocumentRecord(
        id=2,
        term="Ancak",
        culture="Tengger",
        semantic_domain="Material Culture",
        definition="A square-shaped offering container made of woven bamboo.",
        variants="",
    )

    doc3 = DocumentRecord(
        id=3,
        term="Barikan",
        culture="Tengger",
        semantic_domain="Ritual",
        definition="A Barikan is Tengger's protection rite for communal harmony.",
        variants="selamatan\nruwatan\nkenduren\ntolak bala",
    )

    index.build_index([doc1, doc2, doc3])

    # 1. Search for specific term
    results = index.search(query="Among-among", top_k=5)
    assert len(results) > 0
    assert results[0].document.id == 1
    assert "among-among" in results[0].matched_terms

    # 2. Search by variant synonym: "slametan bayi"
    results_variant = index.search(query="slametan bayi", top_k=5)
    assert len(results_variant) > 0
    assert results_variant[0].document.id == 1

    # 3. Search generic term "protection rite"
    results_rite = index.search(query="protection rite", top_k=5)
    # Both Among-among (doc1) and Barikan (doc3) mention "protection rite"
    top_ids = [r.document.id for r in results_rite]
    assert 1 in top_ids
    assert 3 in top_ids
    assert 2 not in top_ids  # Ancak does not mention protection rite

    # 4. Culture and domain filtering
    results_filtered = index.search(
        query="offering",
        top_k=5,
        semantic_domain="Material Culture"
    )
    assert len(results_filtered) == 1
    assert results_filtered[0].document.id == 2


def test_bm25_empty_query():
    index = BM25InvertedIndex()
    index.build_index([])
    results = index.search("")
    assert results == []
