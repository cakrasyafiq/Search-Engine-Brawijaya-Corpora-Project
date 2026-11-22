import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db.session import SessionLocal
from app.services.search_service import search_service
from app.services.semantic_service import semantic_service
from app.services.bge_sparse_service import bge_sparse_service
from evaluation.dataset import load_and_split_dataset
from evaluation.alpha_optimizer import AlphaOptimizer
from evaluation.evaluator import SearchEvaluator
from evaluation.report import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_evaluation")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Information Retrieval and Computational Cost Evaluation"
    )
    parser.add_argument(
        "--source-csv",
        type=str,
        default="Dataset/Evaluation/Evaluation.csv",
        help="Path to evaluation dataset CSV.",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="Dataset/Evaluation/validation_set.csv",
        help="Path to save/load validation set CSV.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="Dataset/Evaluation/test_set.csv",
        help="Path to save/load test set CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Dataset/Evaluation/results",
        help="Directory to save evaluation reports and metrics.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Rank depth cutoff (k=5).",
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.05,
        help="Step size for grid search alpha in [0.0, 1.0].",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.80,
        help="Proportion of queries for test set (default: 0.80, i.e. 80%% test, 20%% val).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified train/test split.",
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="Force re-generation of validation and test split files.",
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=None,
        help="Bypass alpha tuning and use a fixed alpha weight.",
    )
    parser.add_argument(
        "--bcp-csv",
        type=str,
        default=None,
        help="Optional path to external BCP baseline results CSV (query_id,response(k=5)).",
    )
    parser.add_argument(
        "--include-bge-sparse",
        action="store_true",
        help="Also evaluate BGE-M3 Sparse and BGE-M3 Hybrid (Experiment 2 mode).",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        # Step 1: Ensure services and models are primed
        logger.info("Initializing search services...")
        search_service.initialize(db)
        _ = semantic_service.get_model()
        if args.include_bge_sparse:
            _ = semantic_service.get_sparse_linear()
            bge_sparse_service.initialize(db)

        # Step 2: Split Dataset into Validation (20%) and Test (80%)
        logger.info(
            f"Loading and splitting dataset: {args.source_csv} (seed={args.seed}, test_ratio={args.test_ratio})..."
        )
        val_dataset, test_dataset = load_and_split_dataset(
            source_csv=args.source_csv,
            val_output_csv=args.val_csv,
            test_output_csv=args.test_csv,
            test_ratio=args.test_ratio,
            random_state=args.seed,
            force_resplit=args.force_resplit,
        )

        val_type_counts = {
            "Conceptual": len(val_dataset.conceptual_queries),
            "Exact term": len(val_dataset.exact_term_queries),
        }
        if val_dataset.hybrid_queries:
            val_type_counts["Hybrid"] = len(val_dataset.hybrid_queries)

        test_type_counts = {
            "Conceptual": len(test_dataset.conceptual_queries),
            "Exact term": len(test_dataset.exact_term_queries),
        }
        if test_dataset.hybrid_queries:
            test_type_counts["Hybrid"] = len(test_dataset.hybrid_queries)

        logger.info(
            f"Validation set: {len(val_dataset)} queries ({val_type_counts})"
        )
        logger.info(
            f"Test set: {len(test_dataset)} queries ({test_type_counts})"
        )

        # Step 3: Alpha Optimization on Validation Set
        if args.fixed_alpha is not None:
            optimal_alpha = float(args.fixed_alpha)
            optimal_bge_alpha = float(args.fixed_alpha)
            val_records = []
            bge_val_records = []
            logger.info(f"Using user-specified fixed alpha: {optimal_alpha}")
        else:
            logger.info("Starting grid search for optimal alpha (BM25 + BGE-M3) on validation set...")
            optimizer = AlphaOptimizer(
                db=db,
                val_dataset=val_dataset,
                k=args.k,
                alpha_step=args.alpha_step,
                sparse_mode="bm25",
            )
            bm25_prefix = "bm25_alpha_tuning_results" if args.include_bge_sparse else "alpha_tuning_results"
            optimal_alpha, val_records = optimizer.optimize(
                output_dir=out_dir,
                filename_prefix=bm25_prefix,
            )

            bge_val_records = []
            optimal_bge_alpha = optimal_alpha
            if args.include_bge_sparse:
                logger.info("Starting grid search for optimal alpha (BGE-M3 Sparse + BGE-M3 Dense) on validation set...")
                bge_optimizer = AlphaOptimizer(
                    db=db,
                    val_dataset=val_dataset,
                    k=args.k,
                    alpha_step=args.alpha_step,
                    sparse_mode="bge_sparse",
                )
                optimal_bge_alpha, bge_val_records = bge_optimizer.optimize(
                    output_dir=out_dir,
                    filename_prefix="alpha_tuning_results",
                )

        # Step 4: Comprehensive Benchmark on Test Set
        logger.info(
            f"Evaluating search methods on Test Set (BM25 hybrid alpha={optimal_alpha})..."
        )
        evaluator = SearchEvaluator(db=db, k=args.k)
        reports = {}

        if args.bcp_csv:
            logger.info(f"Evaluating external BCP Baseline from {args.bcp_csv}...")
            reports["BCP Lexical Search (Baseline)"] = evaluator.evaluate_external_bcp(
                bcp_csv_path=args.bcp_csv,
                dataset=test_dataset,
                dataset_name="Test Set",
            )

        reports.update(
            evaluator.evaluate_all(
                dataset=test_dataset,
                dataset_name="Test Set",
                optimal_alpha=optimal_alpha,
            )
        )

        # Also evaluate balanced hybrid (alpha=0.6) if optimal_alpha differs
        if abs(optimal_alpha - 0.6) > 0.01:
            logger.info("Evaluating Balanced Hybrid Search (alpha=0.60) for comparison...")
            reports["Hybrid Search (alpha=0.60)"] = evaluator.evaluate_method(
                method_name="Hybrid Search",
                dataset=test_dataset,
                dataset_name="Test Set",
                alpha=0.60,
            )

        if args.include_bge_sparse:
            # Evaluate BGE-M3 Native Sparse Search
            logger.info("Evaluating BGE-M3 Sparse Search on Test Set...")
            reports["BGE-M3 Sparse"] = evaluator.evaluate_method(
                method_name="BGE-M3 Sparse",
                dataset=test_dataset,
                dataset_name="Test Set",
            )

            # Evaluate Full BGE-M3 Hybrid Search (optimal alpha)
            bge_hybrid_name = f"BGE-M3 Hybrid (alpha={optimal_bge_alpha})"
            logger.info(f"Evaluating {bge_hybrid_name} on Test Set...")
            reports[bge_hybrid_name] = evaluator.evaluate_method(
                method_name="BGE-M3 Hybrid",
                dataset=test_dataset,
                dataset_name="Test Set",
                alpha=optimal_bge_alpha,
            )

            # Also evaluate balanced BGE-M3 Hybrid (alpha=0.60) if optimal_bge_alpha differs
            if abs(optimal_bge_alpha - 0.6) > 0.01:
                logger.info("Evaluating Balanced BGE-M3 Hybrid (alpha=0.60) for comparison...")
                reports["BGE-M3 Hybrid (alpha=0.60)"] = evaluator.evaluate_method(
                    method_name="BGE-M3 Hybrid",
                    dataset=test_dataset,
                    dataset_name="Test Set",
                    alpha=0.60,
                )

        # Step 5: Export Metrics and Reports
        report_alpha = optimal_bge_alpha if args.include_bge_sparse else optimal_alpha
        report_val_records = bge_val_records if args.include_bge_sparse else val_records
        ReportGenerator.print_terminal_summary(reports, report_alpha)
        ReportGenerator.export_csv_and_json(reports, out_dir)

        report_md_path = out_dir / "evaluation_report.md"
        ReportGenerator.generate_markdown_report(
            reports=reports,
            optimal_alpha=report_alpha,
            val_records=report_val_records,
            val_query_count=len(val_dataset),
            test_query_count=len(test_dataset),
            output_file=report_md_path,
            val_type_counts=val_type_counts,
            test_type_counts=test_type_counts,
        )
        logger.info(f"Evaluation completed successfully! Reports written to: {out_dir}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
