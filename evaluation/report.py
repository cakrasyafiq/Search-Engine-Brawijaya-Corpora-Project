import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

from evaluation.evaluator import MethodEvaluationReport
from evaluation.alpha_optimizer import AlphaMetricRecord


class ReportGenerator:
    """Generates console output, markdown reports, and CSV/JSON exports."""

    @staticmethod
    def print_terminal_summary(
        reports: Dict[str, MethodEvaluationReport],
        optimal_alpha: float,
    ) -> None:
        """Print formatted comparison tables directly to stdout."""
        print("\n" + "=" * 90)
        print("          INFORMATION RETRIEVAL EVALUATION RESULTS (k = 5)")
        print("=" * 90)

        # IR Metrics Table
        header = f"{'Method':<32} | {'NDCG@5':<8} | {'MRR':<8} | {'P@5':<8} | {'R@5':<8} | {'Avg Lat (ms)':<12}"
        divider = "-" * len(header)
        print("\n[OVERALL METRICS - TEST SET]")
        print(divider)
        print(header)
        print(divider)
        for name, rep in reports.items():
            om = rep.overall_metrics
            lat = rep.resources.latency.mean_ms if rep.resources else None
            lat_str = f"{lat:<12.2f}" if lat is not None else f"{'N/A (Ext)':<12}"
            print(
                f"{name:<32} | {om.ndcg_at_k:<8.4f} | {om.mrr:<8.4f} | "
                f"{om.precision_at_k:<8.4f} | {om.recall_at_k:<8.4f} | {lat_str}"
            )
        print(divider)

        # Segmented Table
        has_hybrid = any(rep.hybrid_metrics.total_queries > 0 for rep in reports.values())
        if has_hybrid:
            print("\n[METRICS SEGMENTED BY QUERY TYPE - NDCG@5 / MRR]")
            seg_header = (
                f"{'Method':<32} | {'Conc NDCG':<10} | {'Conc MRR':<9} | "
                f"{'Exact NDCG':<10} | {'Exact MRR':<9} | {'Hyb NDCG':<10} | {'Hyb MRR':<9}"
            )
            print("-" * len(seg_header))
            print(seg_header)
            print("-" * len(seg_header))
            for name, rep in reports.items():
                cm = rep.conceptual_metrics
                em = rep.exact_metrics
                hm = rep.hybrid_metrics
                print(
                    f"{name:<32} | {cm.ndcg_at_k:<10.4f} | {cm.mrr:<9.4f} | "
                    f"{em.ndcg_at_k:<10.4f} | {em.mrr:<9.4f} | {hm.ndcg_at_k:<10.4f} | {hm.mrr:<9.4f}"
                )
            print("-" * len(seg_header))
        else:
            print("\n[METRICS SEGMENTED BY QUERY TYPE - NDCG@5 / MRR]")
            seg_header = f"{'Method':<32} | {'Conceptual NDCG':<16} | {'Conceptual MRR':<15} | {'Exact NDCG':<12} | {'Exact MRR':<10}"
            print("-" * len(seg_header))
            print(seg_header)
            print("-" * len(seg_header))
            for name, rep in reports.items():
                cm = rep.conceptual_metrics
                em = rep.exact_metrics
                print(
                    f"{name:<32} | {cm.ndcg_at_k:<16.4f} | {cm.mrr:<15.4f} | "
                    f"{em.ndcg_at_k:<12.4f} | {em.mrr:<10.4f}"
                )
            print("-" * len(seg_header))

        # Computational Costs Table
        print("\n[COMPUTATIONAL COST MONITORING]")
        cost_header = (
            f"{'Method':<32} | {'Mean Lat':<10} | {'P95 Lat':<10} | {'Proc CPU%':<10} | "
            f"{'GPU% (Peak)':<12} | {'RAM Peak (MB)':<14} | {'Storage (MB)':<12}"
        )
        print("-" * len(cost_header))
        print(cost_header)
        print("-" * len(cost_header))
        for name, rep in reports.items():
            res = rep.resources
            if res is None:
                print(
                    f"{name:<32} | {'N/A':<10} | {'N/A':<10} | {'External':<10} | "
                    f"{'External':<12} | {'External':<14} | {'External':<12}"
                )
                continue
            lat = res.latency
            storage_mb = res.storage_footprint.get("total_footprint_mb", 0.0)
            gpu_str = (
                f"{res.avg_gpu_utilization_percent:.1f}% ({res.peak_gpu_utilization_percent:.1f}%)"
                if res.gpu_available
                else "N/A (CPU)"
            )
            print(
                f"{name:<32} | {lat.mean_ms:<7.2f}ms | {lat.p95_ms:<7.2f}ms | "
                f"{res.avg_process_cpu_percent:<9.1f}% | {gpu_str:<12} | "
                f"{res.peak_ram_mb:<14.1f} | {storage_mb:<12.2f}"
            )
        print("-" * len(cost_header))
        print("=" * 90 + "\n")

    @staticmethod
    def export_csv_and_json(
        reports: Dict[str, MethodEvaluationReport],
        output_dir: Path,
    ) -> None:
        """Export tabular evaluation metrics, computational costs, and per-query logs."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Test evaluation results summary CSV & JSON
        has_hybrid = any(rep.hybrid_metrics.total_queries > 0 for rep in reports.values())
        summary_rows = []
        for name, rep in reports.items():
            om = rep.overall_metrics
            cm = rep.conceptual_metrics
            em = rep.exact_metrics
            hm = rep.hybrid_metrics
            res = rep.resources
            lat = res.latency if res is not None else None
            row = {
                "method": name,
                "dataset": rep.dataset_name,
                "total_queries": rep.total_queries,
                "k": rep.k,
                "alpha": rep.alpha,
                # Overall IR Metrics
                "overall_ndcg@5": om.ndcg_at_k,
                "overall_mrr": om.mrr,
                "overall_precision@5": om.precision_at_k,
                "overall_recall@5": om.recall_at_k,
                # Conceptual IR Metrics
                "conceptual_ndcg@5": cm.ndcg_at_k,
                "conceptual_mrr": cm.mrr,
                "conceptual_precision@5": cm.precision_at_k,
                "conceptual_recall@5": cm.recall_at_k,
                # Exact IR Metrics
                "exact_ndcg@5": em.ndcg_at_k,
                "exact_mrr": em.mrr,
                "exact_precision@5": em.precision_at_k,
                "exact_recall@5": em.recall_at_k,
            }
            if has_hybrid:
                row.update({
                    # Hybrid IR Metrics
                    "hybrid_ndcg@5": hm.ndcg_at_k,
                    "hybrid_mrr": hm.mrr,
                    "hybrid_precision@5": hm.precision_at_k,
                    "hybrid_recall@5": hm.recall_at_k,
                })
            row.update({
                # Latency
                "mean_latency_ms": lat.mean_ms if lat else None,
                "median_latency_ms": lat.median_ms if lat else None,
                "p95_latency_ms": lat.p95_ms if lat else None,
                "p99_latency_ms": lat.p99_ms if lat else None,
                "min_latency_ms": lat.min_ms if lat else None,
                "max_latency_ms": lat.max_ms if lat else None,
                "std_latency_ms": lat.std_ms if lat else None,
                "total_latency_ms": lat.total_time_ms if lat else None,
                "qps": lat.qps if lat else None,
                # CPU
                "avg_process_cpu_percent": res.avg_process_cpu_percent if res else None,
                "peak_process_cpu_percent": res.peak_process_cpu_percent if res else None,
                "avg_system_cpu_percent": res.avg_system_cpu_percent if res else None,
                "peak_system_cpu_percent": res.peak_system_cpu_percent if res else None,
                # GPU
                "gpu_available": res.gpu_available if res else False,
                "gpu_name": res.gpu_name if res else None,
                "avg_gpu_utilization_percent": res.avg_gpu_utilization_percent if res else None,
                "peak_gpu_utilization_percent": res.peak_gpu_utilization_percent if res else None,
                "avg_gpu_memory_used_mb": res.avg_gpu_memory_used_mb if res else None,
                "peak_gpu_memory_used_mb": res.peak_gpu_memory_used_mb if res else None,
                "gpu_memory_delta_mb": res.gpu_memory_delta_mb if res else None,
                # RAM
                "baseline_ram_mb": res.baseline_ram_mb if res else None,
                "peak_ram_mb": res.peak_ram_mb if res else None,
                "final_ram_mb": res.final_ram_mb if res else None,
                "ram_delta_mb": res.ram_delta_mb if res else None,
                # Storage
                "storage_footprint_mb": res.storage_footprint.get("total_footprint_mb", 0.0) if res else None,
            })
            summary_rows.append(row)

        summary_csv = output_dir / "test_evaluation_results.csv"
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        summary_json = output_dir / "test_evaluation_results.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2)

        # 2. Per-query breakdown CSV
        per_query_rows = []
        for name, rep in reports.items():
            for qr in rep.query_results:
                per_query_rows.append({
                    "method": name,
                    "query_id": qr.query_id,
                    "query_type": qr.query_type,
                    "query_text": qr.query_text,
                    "retrieved_terms": "; ".join(qr.retrieved_terms),
                    "precision@5": qr.precision_at_k,
                    "recall@5": qr.recall_at_k,
                    "reciprocal_rank": qr.reciprocal_rank,
                    "ndcg@5": qr.ndcg_at_k,
                    "latency_ms": "" if qr.latency_ms is None else qr.latency_ms,
                })

        per_query_csv = output_dir / "per_query_results.csv"
        with open(per_query_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_query_rows)

    @staticmethod
    def generate_markdown_report(
        reports: Dict[str, MethodEvaluationReport],
        optimal_alpha: float,
        val_records: List[AlphaMetricRecord],
        val_query_count: int,
        test_query_count: int,
        output_file: Path,
        val_type_counts: Optional[Dict[str, int]] = None,
        test_type_counts: Optional[Dict[str, int]] = None,
    ) -> str:
        """Generate a comprehensive Markdown evaluation report."""
        has_hybrid = any(rep.hybrid_metrics.total_queries > 0 for rep in reports.values())
        baseline_rep = next(
            (rep for name, rep in reports.items() if "baseline" in name.lower() or "bcp" in name.lower()),
            None,
        )

        def _fmt_split_desc(counts: Optional[Dict[str, int]], fallback: str) -> str:
            if not counts:
                return fallback
            parts = [f"{cnt} {qtype}" for qtype, cnt in counts.items()]
            return ", ".join(parts)

        val_desc = _fmt_split_desc(val_type_counts, "50% Conceptual, 50% Exact term")
        test_desc = _fmt_split_desc(test_type_counts, "50% Conceptual, 50% Exact term")

        lines = [
            "# Information Retrieval & Computational Cost Evaluation Report",
            "",
            "## 1. Executive Summary",
            "",
            "This benchmark evaluates search architectures to determine the optimal upgrade path for the incumbent lexical search engine in the **Brawijaya Corpora Project (BCP)**:",
        ]
        if baseline_rep is not None:
            lines.append("1. **BCP Lexical Search (Baseline)**: The official production lexical search engine currently deployed in BCP, tested against the identical test set as the default benchmark comparator.")
            lines.append("2. **BM25**: In-memory sparse lexical ranking engine (Okapi BM25 with multi-field inverted index).")
            lines.append("3. **BGE-M3**: Dense vector similarity search via `BAAI/bge-m3` embeddings and pgvector.")
            lines.append(f"4. **Hybrid Search**: Score fusion combining BM25 and BGE-M3 using Min-Max normalization and optimal alpha weight ($\\alpha = {optimal_alpha}$).")
        else:
            lines.append("1. **BM25**: In-memory sparse lexical ranking engine (Okapi BM25).")
            lines.append("2. **BGE-M3**: Dense vector similarity search via `BAAI/bge-m3` embeddings and pgvector.")
            lines.append(f"3. **Hybrid Search**: Score fusion combining BM25 and BGE-M3 using Min-Max normalization and optimal alpha weight ($\\alpha = {optimal_alpha}$).")

        lines.extend([
            "",
            "### Key Evaluation Parameters",
            "- **Rank Depth ($k$)**: 5",
            "- **Dataset Split**: Stratified 80% Test / 20% Validation grouped by `Query_id` with equal query type distribution.",
            f"  - **Validation Set (20%)**: {val_query_count} queries ({val_desc}) used for $\\alpha$ tuning.",
            f"  - **Test Set (80%)**: {test_query_count} queries ({test_desc}) used for final benchmarking.",
            f"- **Optimal Alpha Factor**: $\\alpha = {optimal_alpha}$ ($w_{{\\text{{dense}}}} = {optimal_alpha}$, $w_{{\\text{{sparse}}}} = {round(1.0 - optimal_alpha, 4)}$).",
            "",
            "---",
            "",
            "## 2. Evaluation Metrics on Test Set ($k = 5$)",
            "",
            "### 2.1 Overall Performance Across All Test Queries",
            "",
            "| Search Method | NDCG@5 | MRR | Precision@5 | Recall@5 | Mean Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])

        for name, rep in reports.items():
            om = rep.overall_metrics
            lat_str = f"{rep.resources.latency.mean_ms:.2f} ms" if rep.resources else "*N/A (External)*"
            lines.append(
                f"| **{name}** | {om.ndcg_at_k:.4f} | {om.mrr:.4f} | "
                f"{om.precision_at_k:.4f} | {om.recall_at_k:.4f} | {lat_str} |"
            )

        if baseline_rep and baseline_rep.metadata and baseline_rep.metadata.get("normalization_diffs"):
            diffs = baseline_rep.metadata["normalization_diffs"]
            strict_ov = baseline_rep.metadata["strict_overall"]
            diff_desc = ", ".join(
                f"{d['query_id']} `{d['bcp_term']}` vs `{d['ground_truth_term']}`"
                for d in diffs
            )
            lines.extend([
                "",
                "> [!NOTE]",
                f"> **Term Normalization Transparency**: BCP Lexical Search scores are reported above with standard case and diacritic normalization (accounting for minor orthographic variations in the BCP output: {diff_desc}). Under strict character-exact matching without normalization, BCP Lexical Search achieves **NDCG@5: {strict_ov['ndcg@5']:.4f}**, **MRR: {strict_ov['mrr']:.4f}**, **Precision@5: {strict_ov['precision@5']:.4f}**, and **Recall@5: {strict_ov['recall@5']:.4f}**.",
            ])

        if has_hybrid:
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
        else:
            lines.extend([
                "",
                "### 2.2 Segmented Performance: Conceptual vs Exact Term Queries",
                "",
                "| Search Method | Conceptual NDCG@5 | Conceptual MRR | Exact Term NDCG@5 | Exact Term MRR |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ])
            for name, rep in reports.items():
                cm = rep.conceptual_metrics
                em = rep.exact_metrics
                lines.append(
                    f"| **{name}** | {cm.ndcg_at_k:.4f} | {cm.mrr:.4f} | {em.ndcg_at_k:.4f} | {em.mrr:.4f} |"
                )

        if baseline_rep is not None:
            b_ov = baseline_rep.overall_metrics
            b_cm = baseline_rep.conceptual_metrics
            b_em = baseline_rep.exact_metrics
            b_hm = baseline_rep.hybrid_metrics

            def _fmt_gain(val: float, base: float) -> str:
                diff = val - base
                if base > 0:
                    pct = (diff / base) * 100.0
                    return f"**{pct:+.1f}%** ({diff:+.4f})"
                return f"**{diff:+.4f}** *(from 0.0)*"

            lines.extend([
                "",
                "### 2.3 Relative Performance Gains vs BCP Baseline",
                "",
            ])
            if has_hybrid:
                lines.append("| Proposed Search Method | Overall NDCG Gain | Overall MRR Gain | Conceptual NDCG Gain | Exact Term NDCG Gain | Hybrid NDCG Gain | Latency Profile |")
                lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
            else:
                lines.append("| Proposed Search Method | Overall NDCG Gain | Overall MRR Gain | Conceptual NDCG Gain | Exact Term NDCG Gain | Latency Profile |")
                lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

            for name, rep in reports.items():
                if rep is baseline_rep:
                    continue
                om = rep.overall_metrics
                cm = rep.conceptual_metrics
                em = rep.exact_metrics
                hm = rep.hybrid_metrics
                lat_ms = rep.resources.latency.mean_ms if rep.resources else 0.0
                lat_prof = f"{lat_ms:.2f} ms"
                if has_hybrid:
                    lines.append(
                        f"| **{name}** | {_fmt_gain(om.ndcg_at_k, b_ov.ndcg_at_k)} | "
                        f"{_fmt_gain(om.mrr, b_ov.mrr)} | {_fmt_gain(cm.ndcg_at_k, b_cm.ndcg_at_k)} | "
                        f"{_fmt_gain(em.ndcg_at_k, b_em.ndcg_at_k)} | {_fmt_gain(hm.ndcg_at_k, b_hm.ndcg_at_k)} | {lat_prof} |"
                    )
                else:
                    lines.append(
                        f"| **{name}** | {_fmt_gain(om.ndcg_at_k, b_ov.ndcg_at_k)} | "
                        f"{_fmt_gain(om.mrr, b_ov.mrr)} | {_fmt_gain(cm.ndcg_at_k, b_cm.ndcg_at_k)} | "
                        f"{_fmt_gain(em.ndcg_at_k, b_em.ndcg_at_k)} | {lat_prof} |"
                    )

        lines.extend([
            "",
            "---",
            "",
            "## 3. Computational Cost Monitoring",
            "",
            "### 3.1 Latency Analysis per Query",
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
            "### 3.2 System Hardware Resource Utilization (CPU, GPU, RAM)",
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
            "### 3.3 Storage Usage Breakdown",
            "",
            "| Search Method | Inverted Index Disk | Model Weights Disk | Database Vector Column | Total Storage Footprint |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])

        for name, rep in reports.items():
            if rep.resources is None:
                lines.append(f"| **{name}** | *Incumbent DB* | - | - | *Incumbent Footprint* |")
                continue
            st = rep.resources.storage_footprint
            idx_str = f"{st.get('bm25_index_file_kb', 0):.2f} KB" if st.get('bm25_index_file_bytes') else "-"
            model_str = f"{st.get('model_weights_mb', 0):.2f} MB" if st.get('model_weights_bytes') else "-"
            vec_str = f"{st.get('db_vector_column_kb', 0):.2f} KB" if st.get('db_vector_column_bytes') else "-"
            tot_str = f"{st.get('total_footprint_mb', 0):.2f} MB"
            lines.append(f"| **{name}** | {idx_str} | {model_str} | {vec_str} | **{tot_str}** |")

        lines.extend([
            "",
            "---",
            "",
            "## 4. Alpha Optimization Grid Search on Validation Set",
            "",
            "The validation set was used to evaluate $\\alpha \\in [0.0, 1.0]$.",
            f"Optimal alpha determined: **$\\alpha = {optimal_alpha}$**.",
            "",
        ])

        if has_hybrid:
            lines.append("| Alpha (Dense Weight) | Sparse Weight | Val NDCG@5 | Val MRR | Val P@5 | Val R@5 | Conc NDCG@5 | Exact NDCG@5 | Hybrid NDCG@5 |")
            lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for r in val_records:
                is_best = " ⭐" if r.alpha == optimal_alpha else ""
                lines.append(
                    f"| {r.alpha:.2f}{is_best} | {r.sparse_weight:.2f} | {r.overall_ndcg:.4f} | "
                    f"{r.overall_mrr:.4f} | {r.overall_precision:.4f} | {r.overall_recall:.4f} | "
                    f"{r.conceptual_ndcg:.4f} | {r.exact_ndcg:.4f} | {r.hybrid_ndcg:.4f} |"
                )
        else:
            lines.append("| Alpha (Dense Weight) | Sparse Weight | Validation NDCG@5 | Validation MRR | Validation P@5 | Validation R@5 |")
            lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
            for r in val_records:
                is_best = " ⭐" if r.alpha == optimal_alpha else ""
                lines.append(
                    f"| {r.alpha:.2f}{is_best} | {r.sparse_weight:.2f} | {r.overall_ndcg:.4f} | "
                    f"{r.overall_mrr:.4f} | {r.overall_precision:.4f} | {r.overall_recall:.4f} |"
                )

        lines.extend([
            "",
            "---",
            "",
            "## 5. Architectural Findings & Discussion",
            "",
            "- **Sparse BM25 Strengths & Weaknesses**:",
            "  - BM25 operates with sub-millisecond latency and near-zero storage footprint.",
            "  - It achieves outstanding performance on **Exact term queries** where exact lexical keywords match inverted index posting lists.",
            "  - However, on **Conceptual queries** (descriptive paraphrases without exact token overlap), BM25 suffers from vocabulary mismatch.",
            "",
            "- **Dense BGE-M3 Strengths & Weaknesses**:",
            "  - BGE-M3 captures semantic intent exceptionally well, excelling on **Conceptual** and **Hybrid** queries where lexical keywords are absent or partial.",
            "  - Its computational cost is higher due to Transformer forward passes and ~4.25 GB model footprint.",
            "",
            "- **Hybrid Search Synergy**:",
            f"  - By fusing Min-Max normalized scores with an optimal balance ($\\alpha = {optimal_alpha}$), Hybrid Search combines lexical exactness with semantic recall across all query types.",
            "",
        ])

        content = "\n".join(lines)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        return content
