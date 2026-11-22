import argparse
import csv
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db.session import SessionLocal
from app.services.search_service import search_service
from app.services.semantic_service import semantic_service
from app.services.adaptive_hybrid_service import adaptive_hybrid_service
from evaluation.dataset import EvaluationDataset, load_and_split_dataset
from evaluation.adaptive_optimizer import AdaptiveGateOptimizer, AdaptiveGateRecord
from evaluation.evaluator import SearchEvaluator, MethodEvaluationReport
from evaluation.metrics import compute_query_metrics, aggregate_metrics, QueryEvaluationResult
from evaluation.report import ReportGenerator
from evaluation.resource_monitor import ResourceMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment_4")


def evaluate_adaptive_hybrid(
    db,
    dataset: EvaluationDataset,
    tau: float,
    scale: float,
    alpha_min: float,
    alpha_max: float = 1.0,
    k: int = 5,
    dataset_name: str = "Test Set",
    warmup: bool = True,
) -> Tuple[MethodEvaluationReport, List[Dict[str, Any]]]:
    """
    Evaluate Adaptive Hybrid Search (Smooth Sigmoid Lexical Confidence Gate)
    across all test queries with full ResourceMonitor instrumentation and
    per-query gate diagnostics.
    """
    method_label = "Adaptive Hybrid Search (Sigmoid Gate)"
    logger.info(
        f"Evaluating {method_label} (tau={tau}, scale={scale}, "
        f"alpha_min={alpha_min}, alpha_max={alpha_max}) on {dataset_name} ({len(dataset)} queries)..."
    )

    if not search_service.is_ready:
        search_service.initialize(db)
    _ = semantic_service.get_model()

    if warmup:
        _ = adaptive_hybrid_service.search(
            db=db,
            query="warmup query",
            top_k=k,
            tau=tau,
            scale=scale,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )

    monitor = ResourceMonitor(db=db)
    monitor.start_monitoring()

    query_results: List[QueryEvaluationResult] = []
    conceptual_results: List[QueryEvaluationResult] = []
    exact_results: List[QueryEvaluationResult] = []
    hybrid_results: List[QueryEvaluationResult] = []
    latencies_ms: List[float] = []
    gate_logs: List[Dict[str, Any]] = []

    for q in dataset.queries:
        t0 = time.perf_counter()
        candidates, eff_dense_w, eff_sparse_w, max_bm25, confidence = adaptive_hybrid_service.search(
            db=db,
            query=q.query_text,
            top_k=k,
            tau=tau,
            scale=scale,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        lat_ms = (time.perf_counter() - t0) * 1000.0
        terms = [c.term for c in candidates[:k]]

        latencies_ms.append(lat_ms)
        res = compute_query_metrics(
            query_id=q.query_id,
            query_type=q.query_type,
            query_text=q.query_text,
            retrieved_terms=terms,
            ground_truth=q.ground_truth_labels,
            latency_ms=lat_ms,
            k=k,
        )
        query_results.append(res)

        qt_low = q.query_type.lower()
        if qt_low == "conceptual":
            conceptual_results.append(res)
        elif qt_low == "hybrid":
            hybrid_results.append(res)
        else:
            exact_results.append(res)

        gate_logs.append({
            "query_id": q.query_id,
            "query_type": q.query_type,
            "query_text": q.query_text,
            "max_bm25_score": max_bm25,
            "lexical_confidence": confidence,
            "adaptive_alpha": eff_dense_w,
            "sparse_weight": eff_sparse_w,
            "precision@5": res.precision_at_k,
            "recall@5": res.recall_at_k,
            "reciprocal_rank": res.reciprocal_rank,
            "ndcg@5": res.ndcg_at_k,
            "latency_ms": res.latency_ms,
            "retrieved_terms": "; ".join(terms),
        })

    resources = monitor.stop_monitoring(
        latencies_ms=latencies_ms,
        method_name="Hybrid Search",
    )

    report = MethodEvaluationReport(
        method_name=method_label,
        dataset_name=dataset_name,
        total_queries=len(dataset),
        k=k,
        alpha=None,
        overall_metrics=aggregate_metrics(query_results),
        conceptual_metrics=aggregate_metrics(conceptual_results),
        exact_metrics=aggregate_metrics(exact_results),
        hybrid_metrics=aggregate_metrics(hybrid_results),
        resources=resources,
        query_results=query_results,
        metadata={
            "tau": tau,
            "scale": scale,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
        },
    )
    return report, gate_logs


def export_adaptive_diagnostics(
    gate_logs: List[Dict[str, Any]],
    reports: Dict[str, MethodEvaluationReport],
    output_dir: Path,
) -> None:
    """Export per-query adaptive alpha(q) diagnostics compared against Static Hybrid."""
    static_a1 = {
        qr.query_id: qr.ndcg_at_k
        for qr in reports["Hybrid Search (Static alpha=1.0)"].query_results
    }
    static_a06 = {
        qr.query_id: qr.ndcg_at_k
        for qr in reports["Hybrid Search (Static alpha=0.60)"].query_results
    }

    enriched: List[Dict[str, Any]] = []
    for row in gate_logs:
        qid = row["query_id"]
        a1_ndcg = static_a1.get(qid, 0.0)
        a06_ndcg = static_a06.get(qid, 0.0)
        enriched.append({
            "query_id": qid,
            "query_type": row["query_type"],
            "query_text": row["query_text"],
            "max_bm25_score": row["max_bm25_score"],
            "lexical_confidence": row["lexical_confidence"],
            "adaptive_alpha": row["adaptive_alpha"],
            "sparse_weight": row["sparse_weight"],
            "adaptive_ndcg@5": row["ndcg@5"],
            "static_a1_ndcg@5": a1_ndcg,
            "static_a06_ndcg@5": a06_ndcg,
            "delta_vs_a1": round(row["ndcg@5"] - a1_ndcg, 4),
            "delta_vs_a06": round(row["ndcg@5"] - a06_ndcg, 4),
            "adaptive_mrr": row["reciprocal_rank"],
            "adaptive_p@5": row["precision@5"],
            "adaptive_r@5": row["recall@5"],
            "latency_ms": row["latency_ms"],
            "retrieved_terms": row["retrieved_terms"],
        })

    diag_path = output_dir / "adaptive_per_query_diagnostics.csv"
    with open(diag_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(enriched[0].keys()))
        writer.writeheader()
        writer.writerows(enriched)


def generate_experiment_4_report(
    reports: Dict[str, MethodEvaluationReport],
    optimal_params: Dict[str, float],
    val_records: List[AdaptiveGateRecord],
    gate_logs: List[Dict[str, Any]],
    val_count: int,
    test_count: int,
    val_type_counts: Dict[str, int],
    test_type_counts: Dict[str, int],
    output_file: Path,
) -> None:
    """Generate a comprehensive Markdown report for Experiment 4."""
    baseline_rep = reports["BCP Lexical Search (Baseline)"]
    static_a1_rep = reports["Hybrid Search (Static alpha=1.0)"]
    adaptive_rep = reports["Adaptive Hybrid Search (Sigmoid Gate)"]

    tau = optimal_params["tau"]
    scale = optimal_params["scale"]
    alpha_min = optimal_params["alpha_min"]
    alpha_max = optimal_params["alpha_max"]

    def _fmt_gain(val: float, base: float) -> str:
        diff = val - base
        if base > 0:
            pct = (diff / base) * 100.0
            return f"**{pct:+.1f}%** ({diff:+.4f})"
        return f"**{diff:+.4f}** *(from 0.0)*"

    val_desc = ", ".join(f"{v} {k}" for k, v in val_type_counts.items())
    test_desc = ", ".join(f"{v} {k}" for k, v in test_type_counts.items())

    lines = [
        "# Experiment 4: Adaptive Hybrid Search (Smooth Sigmoid Lexical Confidence Gate) Evaluation Report",
        "",
        "## 1. Executive Summary",
        "",
        "Experiment 4 evaluates a **Query-Adaptive Hybrid Search** architecture (`GET /search/adaptive-hybrid`) using a **Smooth Sigmoid Lexical Confidence Gate** over the identical `eval_v2` 20/80 stratified split from Experiment 3, comparing against all static architectures and the incumbent BCP baseline:",
        "1. **BCP Lexical Search (Baseline)**: Current production lexical engine evaluated on the 60-query test set.",
        "2. **BM25**: In-memory Okapi BM25 sparse lexical search.",
        "3. **BGE-M3**: Dense 1024-d semantic vector search via `BAAI/bge-m3` + `pgvector`.",
        "4. **Hybrid Search (Static alpha=1.0)**: Experiment 3 validation-optimal static hybrid configuration.",
        "5. **Hybrid Search (Static alpha=0.60)**: Balanced static hybrid configuration.",
        "6. **Adaptive Hybrid Search (Sigmoid Gate)**: Query-adaptive weighted score fusion computing $\\alpha(q)$ dynamically per query from raw BM25 confidence $s_{\\max}(q)$.",
        "",
        "### Gate Formulation & Key Parameters",
        "$$c(q) = \\begin{cases} 0.0, & s_{\\max}(q) \\le 0 \\\\ \\dfrac{1}{1 + \\exp\\!\\left(-\\dfrac{s_{\\max}(q) - \\tau}{s}\\right)}, & s_{\\max}(q) > 0 \\end{cases}, \\qquad \\alpha(q) = \\alpha_{\\max} - (\\alpha_{\\max} - \\alpha_{\\min}) \\cdot c(q)$$",
        "",
        "- **Rank Depth ($k$)**: 5",
        f"- **Validation Set (20%)**: {val_count} queries ({val_desc}) used strictly for gate tuning.",
        f"- **Test Set (80%)**: {test_count} queries ({test_desc}) used for independent benchmarking.",
        f"- **Validation-Optimal Gate Parameters**: $\\tau = {tau}$, $s = {scale}$, $\\alpha_{{\\min}} = {alpha_min}$, $\\alpha_{{\\max}} = {alpha_max}$.",
        "",
        "---",
        "",
        "## 2. Evaluation Metrics on Test Set ($k = 5$)",
        "",
        "### 2.1 Overall Performance Across All Test Queries",
        "",
        "| Search Method | NDCG@5 | MRR | Precision@5 | Recall@5 | Mean Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for name, rep in reports.items():
        om = rep.overall_metrics
        lat_str = f"{rep.resources.latency.mean_ms:.2f} ms" if rep.resources else "*N/A (External)*"
        lines.append(
            f"| **{name}** | {om.ndcg_at_k:.4f} | {om.mrr:.4f} | "
            f"{om.precision_at_k:.4f} | {om.recall_at_k:.4f} | {lat_str} |"
        )

    if baseline_rep.metadata and baseline_rep.metadata.get("normalization_diffs"):
        diffs = baseline_rep.metadata["normalization_diffs"]
        strict_ov = baseline_rep.metadata["strict_overall"]
        diff_desc = ", ".join(
            f"{d['query_id']} `{d['bcp_term']}` vs `{d['ground_truth_term']}`"
            for d in diffs
        )
        lines.extend([
            "",
            "> [!NOTE]",
            f"> **Term Normalization Transparency**: BCP Lexical Search scores are reported with standard case and diacritic normalization ({diff_desc}). Under strict character-exact matching without normalization, BCP achieves **NDCG@5: {strict_ov['ndcg@5']:.4f}**, **MRR: {strict_ov['mrr']:.4f}**, **Precision@5: {strict_ov['precision@5']:.4f}**, and **Recall@5: {strict_ov['recall@5']:.4f}**.",
        ])

    lines.extend([
        "",
        "### 2.2 Segmented Performance: Conceptual vs Exact Term vs Hybrid Queries",
        "",
        "| Search Method | Conceptual NDCG@5 | Conceptual MRR | Exact Term NDCG@5 | Exact Term MRR | Hybrid NDCG@5 | Hybrid MRR |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for name, rep in reports.items():
        cm = rep.conceptual_metrics
        em = rep.exact_metrics
        hm = rep.hybrid_metrics
        lines.append(
            f"| **{name}** | {cm.ndcg_at_k:.4f} | {cm.mrr:.4f} | "
            f"{em.ndcg_at_k:.4f} | {em.mrr:.4f} | {hm.ndcg_at_k:.4f} | {hm.mrr:.4f} |"
        )

    lines.extend([
        "",
        "### 2.3 Relative Performance Gains vs BCP Baseline",
        "",
        "| Proposed Search Method | Overall NDCG Gain | Overall MRR Gain | Conceptual NDCG Gain | Exact Term NDCG Gain | Hybrid NDCG Gain |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for name, rep in reports.items():
        if rep is baseline_rep:
            continue
        om, cm, em, hm = (
            rep.overall_metrics,
            rep.conceptual_metrics,
            rep.exact_metrics,
            rep.hybrid_metrics,
        )
        b_om, b_cm, b_em, b_hm = (
            baseline_rep.overall_metrics,
            baseline_rep.conceptual_metrics,
            baseline_rep.exact_metrics,
            baseline_rep.hybrid_metrics,
        )
        lines.append(
            f"| **{name}** | {_fmt_gain(om.ndcg_at_k, b_om.ndcg_at_k)} | "
            f"{_fmt_gain(om.mrr, b_om.mrr)} | {_fmt_gain(cm.ndcg_at_k, b_cm.ndcg_at_k)} | "
            f"{_fmt_gain(em.ndcg_at_k, b_em.ndcg_at_k)} | {_fmt_gain(hm.ndcg_at_k, b_hm.ndcg_at_k)} |"
        )

    lines.extend([
        "",
        "### 2.4 Adaptive Gate vs Static Hybrid (alpha=1.0 / BGE-M3) Delta",
        "",
        "| Metric Segment | Static Hybrid (alpha=1.0) | Static Hybrid (alpha=0.60) | Adaptive Hybrid (Sigmoid Gate) | Adaptive vs Static alpha=1.0 |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    static_a06_rep = reports["Hybrid Search (Static alpha=0.60)"]
    for label, get_m in [
        ("Overall NDCG@5", lambda r: r.overall_metrics.ndcg_at_k),
        ("Overall MRR", lambda r: r.overall_metrics.mrr),
        ("Conceptual NDCG@5", lambda r: r.conceptual_metrics.ndcg_at_k),
        ("Exact Term NDCG@5", lambda r: r.exact_metrics.ndcg_at_k),
        ("Hybrid NDCG@5", lambda r: r.hybrid_metrics.ndcg_at_k),
    ]:
        v1 = get_m(static_a1_rep)
        v06 = get_m(static_a06_rep)
        va = get_m(adaptive_rep)
        lines.append(
            f"| **{label}** | {v1:.4f} | {v06:.4f} | {va:.4f} | {_fmt_gain(va, v1)} |"
        )

    # Adaptive alpha distribution by query type on test set
    by_qt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in gate_logs:
        by_qt[row["query_type"]].append(row)

    lines.extend([
        "",
        "---",
        "",
        "## 3. Adaptive Gate Behavior on Test Set ($\\alpha(q)$ Distribution)",
        "",
        "| Query Type | Queries | Mean `max_bm25` | Mean Confidence $c(q)$ | Min $\\alpha(q)$ | Mean $\\alpha(q)$ | Max $\\alpha(q)$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for qt in ["Conceptual", "Exact term", "Hybrid"]:
        items = by_qt.get(qt, [])
        if not items:
            continue
        scores = [x["max_bm25_score"] for x in items]
        confs = [x["lexical_confidence"] for x in items]
        alphas = [x["adaptive_alpha"] for x in items]
        lines.append(
            f"| **{qt}** | {len(items)} | {sum(scores)/len(scores):.2f} | "
            f"{sum(confs)/len(confs):.4f} | {min(alphas):.4f} | "
            f"{sum(alphas)/len(alphas):.4f} | {max(alphas):.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Computational Cost Monitoring",
        "",
        "### 4.1 Latency Analysis per Query",
        "",
        "| Search Method | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) | QPS |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for name, rep in reports.items():
        if rep.resources is None:
            lines.append(f"| **{name}** | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |")
            continue
        lat = rep.resources.latency
        lines.append(
            f"| **{name}** | {lat.mean_ms:.2f} | {lat.median_ms:.2f} | {lat.p95_ms:.2f} | "
            f"{lat.min_ms:.2f} | {lat.max_ms:.2f} | {lat.qps:.1f} |"
        )

    lines.extend([
        "",
        "### 4.2 System Hardware Resource Utilization (CPU, GPU, RAM)",
        "",
        "| Search Method | Process CPU Avg (Peak) | System CPU Avg (Peak) | GPU Util Avg (Peak) | GPU VRAM Peak | RAM RSS Peak (Delta) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for name, rep in reports.items():
        res = rep.resources
        if res is None:
            lines.append(f"| **{name}** | *External* | *External* | *External* | *External* | *External* |")
            continue
        gpu_str = (
            f"{res.avg_gpu_utilization_percent:.1f}% ({res.peak_gpu_utilization_percent:.1f}%)"
            if res.gpu_available
            else "N/A (CPU)"
        )
        vram_str = f"{res.peak_gpu_memory_used_mb:.1f} MiB" if res.gpu_available else "N/A"
        lines.append(
            f"| **{name}** | {res.avg_process_cpu_percent:.1f}% ({res.peak_process_cpu_percent:.1f}%) | "
            f"{res.avg_system_cpu_percent:.1f}% ({res.peak_system_cpu_percent:.1f}%) | {gpu_str} | {vram_str} | "
            f"{res.peak_ram_mb:.1f} MB (+{res.ram_delta_mb:.1f} MB) |"
        )

    lines.extend([
        "",
        "### 4.3 Storage Usage Breakdown",
        "",
        "| Search Method | Inverted Index Disk | Model Weights Disk | Database Vector Column | Total Storage Footprint |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for name, rep in reports.items():
        if rep.resources is None:
            lines.append(f"| **{name}** | *Incumbent DB* | - | - | *Incumbent Footprint* |")
            continue
        st = rep.resources.storage_footprint
        idx_str = f"{st.get('bm25_index_file_kb', 0):.2f} KB" if st.get("bm25_index_file_bytes") else "-"
        model_str = f"{st.get('model_weights_mb', 0):.2f} MB" if st.get("model_weights_bytes") else "-"
        vec_str = f"{st.get('db_vector_column_kb', 0):.2f} KB" if st.get("db_vector_column_bytes") else "-"
        tot_str = f"{st.get('total_footprint_mb', 0):.2f} MB"
        lines.append(f"| **{name}** | {idx_str} | {model_str} | {vec_str} | **{tot_str}** |")

    top_10 = sorted(
        val_records,
        key=lambda r: (r.overall_ndcg, r.overall_mrr, r.overall_recall, -abs(r.scale - 1.0), -abs(r.tau - 8.0)),
        reverse=True,
    )[:10]

    lines.extend([
        "",
        "---",
        "",
        "## 5. Validation Grid Search Top Configurations (Sigmoid Gate)",
        "",
        f"Evaluated {len(val_records)} parameter combinations $(\\tau, s, \\alpha_{{\\min}})$ with $\\alpha_{{\\max}} = {alpha_max}$ on the 15-query validation set.",
        "",
        "| Rank | $\\tau$ | $s$ | $\\alpha_{\\min}$ | Val NDCG@5 | Val MRR | Val P@5 | Val R@5 | Conc NDCG@5 | Exact NDCG@5 | Hybrid NDCG@5 |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for idx, r in enumerate(top_10, start=1):
        star = " ⭐" if idx == 1 else ""
        lines.append(
            f"| {idx}{star} | {r.tau:.1f} | {r.scale:.1f} | {r.alpha_min:.2f} | "
            f"{r.overall_ndcg:.4f} | {r.overall_mrr:.4f} | {r.overall_precision:.4f} | "
            f"{r.overall_recall:.4f} | {r.conceptual_ndcg:.4f} | {r.exact_ndcg:.4f} | {r.hybrid_ndcg:.4f} |"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: Adaptive Hybrid Search (Smooth Sigmoid Lexical Confidence Gate)"
    )
    parser.add_argument(
        "--source-csv",
        type=str,
        default="Dataset/Evaluation/eval_v2/Evaluation_v2.csv",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="Dataset/Evaluation/eval_v2/validation_set.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="Dataset/Evaluation/eval_v2/test_set.csv",
    )
    parser.add_argument(
        "--bcp-csv",
        type=str,
        default="Dataset/Evaluation/eval_v2/current_BCP_IR_test_results_v2.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Dataset/Evaluation/results/exp_4",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        logger.info("Initializing search services for Experiment 4...")
        search_service.initialize(db)
        _ = semantic_service.get_model()

        val_dataset, test_dataset = load_and_split_dataset(
            source_csv=args.source_csv,
            val_output_csv=args.val_csv,
            test_output_csv=args.test_csv,
            test_ratio=args.test_ratio,
            random_state=args.seed,
            force_resplit=False,
        )

        val_type_counts = {
            "Conceptual": len(val_dataset.conceptual_queries),
            "Exact term": len(val_dataset.exact_term_queries),
            "Hybrid": len(val_dataset.hybrid_queries),
        }
        test_type_counts = {
            "Conceptual": len(test_dataset.conceptual_queries),
            "Exact term": len(test_dataset.exact_term_queries),
            "Hybrid": len(test_dataset.hybrid_queries),
        }
        logger.info(f"Validation set: {len(val_dataset)} queries ({val_type_counts})")
        logger.info(f"Test set: {len(test_dataset)} queries ({test_type_counts})")

        # 1. Tune Sigmoid Gate strictly on Validation Set
        gate_optimizer = AdaptiveGateOptimizer(
            db=db,
            val_dataset=val_dataset,
            k=args.k,
        )
        optimal_params, val_records = gate_optimizer.optimize(
            output_dir=out_dir,
            filename_prefix="adaptive_gate_tuning_results",
        )

        # 2. Evaluate Benchmark Comparators + Adaptive Hybrid Search on Test Set
        evaluator = SearchEvaluator(db=db, k=args.k)
        reports: Dict[str, MethodEvaluationReport] = {}

        reports["BCP Lexical Search (Baseline)"] = evaluator.evaluate_external_bcp(
            bcp_csv_path=args.bcp_csv,
            dataset=test_dataset,
            dataset_name="Test Set",
        )
        reports["BM25"] = evaluator.evaluate_method(
            method_name="BM25",
            dataset=test_dataset,
            dataset_name="Test Set",
        )
        reports["BGE-M3"] = evaluator.evaluate_method(
            method_name="BGE-M3",
            dataset=test_dataset,
            dataset_name="Test Set",
        )
        reports["Hybrid Search (Static alpha=1.0)"] = evaluator.evaluate_method(
            method_name="Hybrid Search",
            dataset=test_dataset,
            dataset_name="Test Set",
            alpha=1.0,
        )
        reports["Hybrid Search (Static alpha=0.60)"] = evaluator.evaluate_method(
            method_name="Hybrid Search",
            dataset=test_dataset,
            dataset_name="Test Set",
            alpha=0.60,
        )

        adaptive_report, gate_logs = evaluate_adaptive_hybrid(
            db=db,
            dataset=test_dataset,
            tau=optimal_params["tau"],
            scale=optimal_params["scale"],
            alpha_min=optimal_params["alpha_min"],
            alpha_max=optimal_params["alpha_max"],
            k=args.k,
            dataset_name="Test Set",
        )
        reports["Adaptive Hybrid Search (Sigmoid Gate)"] = adaptive_report

        # 3. Export Results & Reports
        ReportGenerator.print_terminal_summary(reports, optimal_params["alpha_min"])
        ReportGenerator.export_csv_and_json(reports, out_dir)
        export_adaptive_diagnostics(gate_logs, reports, out_dir)
        generate_experiment_4_report(
            reports=reports,
            optimal_params=optimal_params,
            val_records=val_records,
            gate_logs=gate_logs,
            val_count=len(val_dataset),
            test_count=len(test_dataset),
            val_type_counts=val_type_counts,
            test_type_counts=test_type_counts,
            output_file=out_dir / "evaluation_report.md",
        )

        logger.info(
            f"Experiment 4 complete! Optimal gate params={optimal_params}. Results saved to: {out_dir}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
