# Experiment 4: Adaptive Hybrid Search (Smooth Sigmoid Lexical Confidence Gate) Evaluation Report

## 1. Executive Summary

Experiment 4 evaluates a **Query-Adaptive Hybrid Search** architecture (`GET /search/adaptive-hybrid`) using a **Smooth Sigmoid Lexical Confidence Gate** over the identical `eval_v2` 20/80 stratified split from Experiment 3, comparing against all static architectures and the incumbent BCP baseline:
1. **BCP Lexical Search (Baseline)**: Current production lexical engine evaluated on the 60-query test set.
2. **BM25**: In-memory Okapi BM25 sparse lexical search.
3. **BGE-M3**: Dense 1024-d semantic vector search via `BAAI/bge-m3` + `pgvector`.
4. **Hybrid Search (Static alpha=1.0)**: Experiment 3 validation-optimal static hybrid configuration.
5. **Hybrid Search (Static alpha=0.60)**: Balanced static hybrid configuration.
6. **Adaptive Hybrid Search (Sigmoid Gate)**: Query-adaptive weighted score fusion computing $\alpha(q)$ dynamically per query from raw BM25 confidence $s_{\max}(q)$.

### Gate Formulation & Key Parameters
$$c(q) = \begin{cases} 0.0, & s_{\max}(q) \le 0 \\ \dfrac{1}{1 + \exp\!\left(-\dfrac{s_{\max}(q) - \tau}{s}\right)}, & s_{\max}(q) > 0 \end{cases}, \qquad \alpha(q) = \alpha_{\max} - (\alpha_{\max} - \alpha_{\min}) \cdot c(q)$$

- **Rank Depth ($k$)**: 5
- **Validation Set (20%)**: 15 queries (5 Conceptual, 5 Exact term, 5 Hybrid) used strictly for gate tuning.
- **Test Set (80%)**: 60 queries (20 Conceptual, 20 Exact term, 20 Hybrid) used for independent benchmarking.
- **Validation-Optimal Gate Parameters**: $\tau = 8.0$, $s = 1.5$, $\alpha_{\min} = 0.45$, $\alpha_{\max} = 1.0$.

---

## 2. Evaluation Metrics on Test Set ($k = 5$)

### 2.1 Overall Performance Across All Test Queries

| Search Method | NDCG@5 | MRR | Precision@5 | Recall@5 | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BCP Lexical Search (Baseline)** | 0.2589 | 0.3250 | 0.1633 | 0.2722 | *N/A (External)* |
| **BM25** | 0.7079 | 0.8033 | 0.4533 | 0.6806 | 0.20 ms |
| **BGE-M3** | 0.8239 | 0.9514 | 0.4733 | 0.7264 | 200.78 ms |
| **Hybrid Search (Static alpha=1.0)** | 0.8239 | 0.9514 | 0.4733 | 0.7264 | 221.56 ms |
| **Hybrid Search (Static alpha=0.60)** | 0.8163 | 0.9444 | 0.4733 | 0.7250 | 213.76 ms |
| **Adaptive Hybrid Search (Sigmoid Gate)** | 0.8398 | 0.9722 | 0.4900 | 0.7486 | 213.31 ms |

> [!NOTE]
> **Term Normalization Transparency**: BCP Lexical Search scores are reported with standard case and diacritic normalization (Q-028 `Prapen` vs `Prapèn`, Q-029 `Jimat Klontongan` vs `Jimat klontongan`). Under strict character-exact matching without normalization, BCP achieves **NDCG@5: 0.2546**, **MRR: 0.3250**, **Precision@5: 0.1567**, and **Recall@5: 0.2625**.

### 2.2 Segmented Performance: Conceptual vs Exact Term vs Hybrid Queries

| Search Method | Conceptual NDCG@5 | Conceptual MRR | Exact Term NDCG@5 | Exact Term MRR | Hybrid NDCG@5 | Hybrid MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BCP Lexical Search (Baseline)** | 0.0000 | 0.0000 | 0.7767 | 0.9750 | 0.0000 | 0.0000 |
| **BM25** | 0.3409 | 0.4350 | 0.9828 | 1.0000 | 0.8000 | 0.9750 |
| **BGE-M3** | 0.8010 | 0.9750 | 0.9016 | 0.9625 | 0.7693 | 0.9167 |
| **Hybrid Search (Static alpha=1.0)** | 0.8010 | 0.9750 | 0.9016 | 0.9625 | 0.7693 | 0.9167 |
| **Hybrid Search (Static alpha=0.60)** | 0.7142 | 0.8833 | 0.9340 | 1.0000 | 0.8007 | 0.9500 |
| **Adaptive Hybrid Search (Sigmoid Gate)** | 0.7636 | 0.9417 | 0.9379 | 1.0000 | 0.8179 | 0.9750 |

### 2.3 Relative Performance Gains vs BCP Baseline

| Proposed Search Method | Overall NDCG Gain | Overall MRR Gain | Conceptual NDCG Gain | Exact Term NDCG Gain | Hybrid NDCG Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25** | **+173.4%** (+0.4490) | **+147.2%** (+0.4783) | **+0.3409** *(from 0.0)* | **+26.5%** (+0.2061) | **+0.8000** *(from 0.0)* |
| **BGE-M3** | **+218.2%** (+0.5650) | **+192.7%** (+0.6264) | **+0.8010** *(from 0.0)* | **+16.1%** (+0.1249) | **+0.7693** *(from 0.0)* |
| **Hybrid Search (Static alpha=1.0)** | **+218.2%** (+0.5650) | **+192.7%** (+0.6264) | **+0.8010** *(from 0.0)* | **+16.1%** (+0.1249) | **+0.7693** *(from 0.0)* |
| **Hybrid Search (Static alpha=0.60)** | **+215.3%** (+0.5574) | **+190.6%** (+0.6194) | **+0.7142** *(from 0.0)* | **+20.3%** (+0.1573) | **+0.8007** *(from 0.0)* |
| **Adaptive Hybrid Search (Sigmoid Gate)** | **+224.4%** (+0.5809) | **+199.1%** (+0.6472) | **+0.7636** *(from 0.0)* | **+20.8%** (+0.1612) | **+0.8179** *(from 0.0)* |

### 2.4 Adaptive Gate vs Static Hybrid (alpha=1.0 / BGE-M3) Delta

| Metric Segment | Static Hybrid (alpha=1.0) | Static Hybrid (alpha=0.60) | Adaptive Hybrid (Sigmoid Gate) | Adaptive vs Static alpha=1.0 |
| :--- | :---: | :---: | :---: | :---: |
| **Overall NDCG@5** | 0.8239 | 0.8163 | 0.8398 | **+1.9%** (+0.0159) |
| **Overall MRR** | 0.9514 | 0.9444 | 0.9722 | **+2.2%** (+0.0208) |
| **Conceptual NDCG@5** | 0.8010 | 0.7142 | 0.7636 | **-4.7%** (-0.0374) |
| **Exact Term NDCG@5** | 0.9016 | 0.9340 | 0.9379 | **+4.0%** (+0.0363) |
| **Hybrid NDCG@5** | 0.7693 | 0.8007 | 0.8179 | **+6.3%** (+0.0486) |

---

## 3. Adaptive Gate Behavior on Test Set ($\alpha(q)$ Distribution)

| Query Type | Queries | Mean `max_bm25` | Mean Confidence $c(q)$ | Min $\alpha(q)$ | Mean $\alpha(q)$ | Max $\alpha(q)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conceptual** | 20 | 5.19 | 0.2202 | 0.4537 | 0.8789 | 1.0000 |
| **Exact term** | 20 | 12.43 | 0.8279 | 0.4501 | 0.5447 | 0.7094 |
| **Hybrid** | 20 | 12.53 | 0.8611 | 0.4501 | 0.5264 | 0.6649 |

---

## 4. Computational Cost Monitoring

### 4.1 Latency Analysis per Query

| Search Method | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) | QPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BCP Lexical Search (Baseline)** | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |
| **BM25** | 0.20 | 0.10 | 0.61 | 0.01 | 0.81 | 4934.3 |
| **BGE-M3** | 200.78 | 198.11 | 307.07 | 141.50 | 351.31 | 5.0 |
| **Hybrid Search (Static alpha=1.0)** | 221.56 | 217.46 | 308.37 | 161.96 | 556.87 | 4.5 |
| **Hybrid Search (Static alpha=0.60)** | 213.76 | 216.25 | 264.58 | 141.48 | 284.91 | 4.7 |
| **Adaptive Hybrid Search (Sigmoid Gate)** | 213.31 | 216.01 | 260.92 | 157.63 | 281.75 | 4.7 |

### 4.2 System Hardware Resource Utilization (CPU, GPU, RAM)

| Search Method | Process CPU Avg (Peak) | System CPU Avg (Peak) | GPU Util Avg (Peak) | GPU VRAM Peak | RAM RSS Peak (Delta) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BCP Lexical Search (Baseline)** | *External* | *External* | *External* | *External* | *External* |
| **BM25** | 0.0% (0.0%) | 0.0% (0.0%) | 6.0% (6.0%) | 1572.0 MiB | 1929.2 MB (+0.0 MB) |
| **BGE-M3** | 546.9% (669.6%) | 67.0% (99.3%) | 7.1% (16.0%) | 1601.0 MiB | 1938.7 MB (+9.4 MB) |
| **Hybrid Search (Static alpha=1.0)** | 537.9% (676.0%) | 68.8% (99.1%) | 11.1% (18.0%) | 1611.0 MiB | 1939.2 MB (+0.9 MB) |
| **Hybrid Search (Static alpha=0.60)** | 559.8% (669.6%) | 67.3% (97.8%) | 7.7% (12.0%) | 1603.0 MiB | 1939.0 MB (+0.7 MB) |
| **Adaptive Hybrid Search (Sigmoid Gate)** | 557.7% (680.8%) | 68.3% (99.3%) | 10.0% (14.0%) | 1594.0 MiB | 1939.0 MB (+0.8 MB) |

### 4.3 Storage Usage Breakdown

| Search Method | Inverted Index Disk | Model Weights Disk | Database Vector Column | Total Storage Footprint |
| :--- | :---: | :---: | :---: | :---: |
| **BCP Lexical Search (Baseline)** | *Incumbent DB* | - | - | *Incumbent Footprint* |
| **BM25** | 338.90 KB | - | - | **0.33 MB** |
| **BGE-M3** | - | 4352.95 MB | 676.66 KB | **4353.61 MB** |
| **Hybrid Search (Static alpha=1.0)** | 338.90 KB | 4352.95 MB | 676.66 KB | **4353.94 MB** |
| **Hybrid Search (Static alpha=0.60)** | 338.90 KB | 4352.95 MB | 676.66 KB | **4353.94 MB** |
| **Adaptive Hybrid Search (Sigmoid Gate)** | 338.90 KB | 4352.95 MB | 676.66 KB | **4353.94 MB** |

---

## 5. Validation Grid Search Top Configurations (Sigmoid Gate)

Evaluated 360 parameter combinations $(\tau, s, \alpha_{\min})$ with $\alpha_{\max} = 1.0$ on the 15-query validation set.

| Rank | $\tau$ | $s$ | $\alpha_{\min}$ | Val NDCG@5 | Val MRR | Val P@5 | Val R@5 | Conc NDCG@5 | Exact NDCG@5 | Hybrid NDCG@5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 ⭐ | 8.0 | 1.5 | 0.45 | 0.8644 | 1.0000 | 0.5200 | 0.7722 | 0.7813 | 0.9689 | 0.8430 |
| 2 | 8.5 | 2.0 | 0.45 | 0.8644 | 1.0000 | 0.5200 | 0.7722 | 0.7813 | 0.9689 | 0.8430 |
| 3 | 9.0 | 2.0 | 0.30 | 0.8644 | 1.0000 | 0.5200 | 0.7722 | 0.7813 | 0.9689 | 0.8430 |
| 4 | 8.0 | 2.0 | 0.45 | 0.8637 | 1.0000 | 0.5200 | 0.7722 | 0.7792 | 0.9689 | 0.8430 |
| 5 | 8.5 | 2.0 | 0.35 | 0.8637 | 1.0000 | 0.5200 | 0.7722 | 0.7792 | 0.9689 | 0.8430 |
| 6 | 8.5 | 2.0 | 0.40 | 0.8637 | 1.0000 | 0.5200 | 0.7722 | 0.7792 | 0.9689 | 0.8430 |
| 7 | 9.0 | 2.0 | 0.25 | 0.8637 | 1.0000 | 0.5200 | 0.7722 | 0.7792 | 0.9689 | 0.8430 |
| 8 | 8.5 | 1.5 | 0.30 | 0.8625 | 1.0000 | 0.5200 | 0.7722 | 0.7813 | 0.9689 | 0.8372 |
| 9 | 9.5 | 3.0 | 0.35 | 0.8574 | 1.0000 | 0.5067 | 0.7500 | 0.7604 | 0.9689 | 0.8430 |
| 10 | 10.0 | 3.0 | 0.25 | 0.8574 | 1.0000 | 0.5067 | 0.7500 | 0.7604 | 0.9689 | 0.8430 |
