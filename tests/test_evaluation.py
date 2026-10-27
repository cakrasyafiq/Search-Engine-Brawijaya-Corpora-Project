import pytest
from pathlib import Path
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    compute_query_metrics,
    aggregate_metrics,
)
from evaluation.dataset import EvaluationDataset, load_and_split_dataset
from evaluation.resource_monitor import StorageAuditor, ResourceSampler


def test_precision_at_k():
    ground_truth = {"A": 2, "B": 1, "C": 1}
    # 2 relevant out of top 5 -> 2/5 = 0.4
    retrieved = ["A", "X", "B", "Y", "Z"]
    assert precision_at_k(retrieved, ground_truth, k=5) == 0.4

    # All 5 relevant -> 5/5 = 1.0
    all_rel = {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1}
    assert precision_at_k(["A", "B", "C", "D", "E"], all_rel, k=5) == 1.0

    # 0 relevant -> 0.0
    assert precision_at_k(["X", "Y", "Z"], ground_truth, k=5) == 0.0


def test_recall_at_k():
    ground_truth = {"A": 2, "B": 1, "C": 1}  # Total 3 relevant
    # 2 retrieved out of 3 total -> 2/3 ≈ 0.6667
    retrieved = ["A", "X", "B", "Y", "Z"]
    assert pytest.approx(recall_at_k(retrieved, ground_truth, k=5), 0.001) == 2.0 / 3.0

    # All 3 retrieved -> 3/3 = 1.0
    retrieved_all = ["A", "B", "C", "X", "Y"]
    assert recall_at_k(retrieved_all, ground_truth, k=5) == 1.0


def test_reciprocal_rank():
    ground_truth = {"Target": 2}
    # Hit at rank 1 -> 1/1 = 1.0
    assert reciprocal_rank(["Target", "B", "C"], ground_truth, k=5) == 1.0
    # Hit at rank 2 -> 1/2 = 0.5
    assert reciprocal_rank(["A", "Target", "C"], ground_truth, k=5) == 0.5
    # Hit at rank 4 -> 1/4 = 0.25
    assert reciprocal_rank(["A", "B", "C", "Target"], ground_truth, k=5) == 0.25
    # No hit within k=5 -> 0.0
    assert reciprocal_rank(["A", "B", "C", "D", "E"], ground_truth, k=5) == 0.0


def test_ndcg_at_k():
    ground_truth = {"A": 2, "B": 1}
    # Perfect ranking: [A (2), B (1)] -> NDCG = 1.0
    assert pytest.approx(ndcg_at_k(["A", "B", "C"], ground_truth, k=5), 0.0001) == 1.0

    # Reversed ranking: [B (1), A (2)] -> strictly < 1.0
    ndcg_reversed = ndcg_at_k(["B", "A", "C"], ground_truth, k=5)
    assert 0.0 < ndcg_reversed < 1.0

    # Completely non-relevant -> 0.0
    assert ndcg_at_k(["X", "Y", "Z"], ground_truth, k=5) == 0.0


def test_dataset_split_proportions(tmp_path):
    source_csv = Path("Dataset/Evaluation/Evaluation.csv")
    if not source_csv.exists():
        pytest.skip("Evaluation.csv not found")

    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"

    val_dataset, test_dataset = load_and_split_dataset(
        source_csv=source_csv,
        val_output_csv=val_csv,
        test_output_csv=test_csv,
        test_ratio=0.80,
        random_state=42,
        force_resplit=True,
    )

    # 50 total queries -> 20% validation is 10, 80% test is 40
    assert len(val_dataset) == 10
    assert len(test_dataset) == 40

    # Check 50/50 balance between query types
    assert len(val_dataset.conceptual_queries) == 5
    assert len(val_dataset.exact_term_queries) == 5
    assert len(test_dataset.conceptual_queries) == 20
    assert len(test_dataset.exact_term_queries) == 20

    # Check no overlap between sets
    val_ids = {q.query_id for q in val_dataset}
    test_ids = {q.query_id for q in test_dataset}
    assert len(val_ids.intersection(test_ids)) == 0


def test_storage_auditor():
    bm25_bytes = StorageAuditor.get_file_size_bytes("bm25_index.pkl")
    assert bm25_bytes > 0

    hf_bytes = StorageAuditor.get_hf_model_size_bytes("BAAI/bge-m3")
    assert hf_bytes >= 0  # May be positive if downloaded
