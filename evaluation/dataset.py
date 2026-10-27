import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class TargetDocument:
    term: str
    definition: str
    relevance_label: int  # 1 (relevant) or 2 (highly relevant)
    rationale: str = ""


@dataclass
class EvaluationQuery:
    query_id: str
    query_type: str  # 'Conceptual' or 'Exact term'
    query_text: str
    targets: Dict[str, TargetDocument] = field(default_factory=dict)

    @property
    def target_terms(self) -> List[str]:
        return list(self.targets.keys())

    @property
    def ground_truth_labels(self) -> Dict[str, int]:
        """Mapping from target term to relevance label (1 or 2)."""
        return {term: doc.relevance_label for term, doc in self.targets.items()}

    @property
    def total_relevant(self) -> int:
        """Total number of relevant documents (label >= 1)."""
        return sum(1 for doc in self.targets.values() if doc.relevance_label >= 1)


class EvaluationDataset:
    """Manages loading, grouping, and accessing queries and ground truth."""

    def __init__(self, queries: List[EvaluationQuery]):
        self.queries = queries
        self._by_id = {q.query_id: q for q in queries}

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)

    def get_query(self, query_id: str) -> Optional[EvaluationQuery]:
        return self._by_id.get(query_id)

    @property
    def conceptual_queries(self) -> List[EvaluationQuery]:
        return [q for q in self.queries if q.query_type.lower() == "conceptual"]

    @property
    def exact_term_queries(self) -> List[EvaluationQuery]:
        return [q for q in self.queries if q.query_type.lower() == "exact term"]

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "EvaluationDataset":
        """Load evaluation queries and ground truth from a CSV file."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Evaluation CSV not found at: {csv_path}")

        query_map: Dict[str, EvaluationQuery] = {}
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row["Query_id"].strip()
                qtype = row["Query_type"].strip()
                qtext = row["Query_text"].strip()
                term = row["Target_Term"].strip()
                definition = row.get("Target_Term_Definition", "").strip()
                
                # Support both Relavance_label and Relevance_label
                label_raw = row.get("Relavance_label") or row.get("Relevance_label") or "1"
                try:
                    label = int(label_raw.strip())
                except ValueError:
                    label = 1

                rationale = row.get("Rationale", "").strip()

                if qid not in query_map:
                    query_map[qid] = EvaluationQuery(
                        query_id=qid,
                        query_type=qtype,
                        query_text=qtext,
                    )

                query_map[qid].targets[term] = TargetDocument(
                    term=term,
                    definition=definition,
                    relevance_label=label,
                    rationale=rationale,
                )

        return cls(list(query_map.values()))

    def to_csv(self, output_path: str | Path) -> None:
        """Export dataset back to CSV format matching Evaluation.csv."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "Query_id",
            "Query_type",
            "Query_text",
            "Target_Term",
            "Target_Term_Definition",
            "Relavance_label",
            "Rationale",
        ]

        with open(output_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for q in self.queries:
                for term, target in q.targets.items():
                    writer.writerow({
                        "Query_id": q.query_id,
                        "Query_type": q.query_type,
                        "Query_text": q.query_text,
                        "Target_Term": term,
                        "Target_Term_Definition": target.definition,
                        "Relavance_label": target.relevance_label,
                        "Rationale": target.rationale,
                    })


def load_and_split_dataset(
    source_csv: str | Path,
    val_output_csv: Optional[str | Path] = None,
    test_output_csv: Optional[str | Path] = None,
    test_ratio: float = 0.80,
    random_state: int = 42,
    force_resplit: bool = False,
) -> Tuple[EvaluationDataset, EvaluationDataset]:
    """
    Split the evaluation dataset into a validation set (20%) and test set (80%)
    with equal query distribution across Conceptual and Exact term queries.

    Returns:
        (validation_dataset, test_dataset)
    """
    source_path = Path(source_csv)
    if val_output_csv is None:
        val_output_csv = source_path.parent / "validation_set.csv"
    else:
        val_output_csv = Path(val_output_csv)

    if test_output_csv is None:
        test_output_csv = source_path.parent / "test_set.csv"
    else:
        test_output_csv = Path(test_output_csv)

    # Check if split files already exist and force_resplit is False
    if not force_resplit and val_output_csv.exists() and test_output_csv.exists():
        logger.info(f"Loading existing split datasets from {val_output_csv} and {test_output_csv}")
        val_dataset = EvaluationDataset.from_csv(val_output_csv)
        test_dataset = EvaluationDataset.from_csv(test_output_csv)
        return val_dataset, test_dataset

    # Perform stratified train/test split on queries
    full_dataset = EvaluationDataset.from_csv(source_path)
    queries = full_dataset.queries
    query_types = [q.query_type for q in queries]

    val_queries, test_queries = train_test_split(
        queries,
        test_size=test_ratio,
        random_state=random_state,
        stratify=query_types,
    )

    # Sort queries by Query_id for neatness
    val_queries.sort(key=lambda q: q.query_id)
    test_queries.sort(key=lambda q: q.query_id)

    val_dataset = EvaluationDataset(val_queries)
    test_dataset = EvaluationDataset(test_queries)

    # Save to disk
    val_dataset.to_csv(val_output_csv)
    test_dataset.to_csv(test_output_csv)
    logger.info(
        f"Dataset split saved: {len(val_dataset)} validation queries to {val_output_csv}, "
        f"{len(test_dataset)} test queries to {test_output_csv}"
    )

    return val_dataset, test_dataset
