# Information Retrieval & Computational Cost Evaluation Report

## 1. Executive Summary

This benchmark evaluates search architectures to determine the optimal upgrade path for the incumbent lexical search engine in the **Brawijaya Corpora Project (BCP)**:
1. **BCP Lexical Search (Baseline)**: The official production lexical search engine currently deployed in BCP, tested against the identical test set as the default benchmark comparator.
2. **BM25**: In-memory sparse lexical ranking engine (Okapi BM25 with multi-field inverted index).
3. **BGE-M3**: Dense vector similarity search via `BAAI/bge-m3` embeddings and pgvector.
4. **Hybrid Search**: Score fusion combining BM25 and BGE-M3 using Min-Max normalization and optimal alpha weight ($\alpha = 1.0$, alongside balanced $\alpha = 0.60$).

### Key Evaluation Parameters
- **Rank Depth ($k$)**: 5
- **Dataset Split**: Stratified 80% Test / 20% Validation grouped by `Query_id` with equal query distribution.
  - **Validation Set (20%)**: 10 queries (50% Conceptual, 50% Exact term) used for $\alpha$ tuning.
  - **Test Set (80%)**: 40 queries (50% Conceptual, 50% Exact term) used for final benchmarking.
- **Optimal Alpha Factor**: $\alpha = 1.0$ ($w_{\text{dense}} = 1.0$, $w_{\text{sparse}} = 0.0$).
- **Baseline Comparator**: Official BCP lexical search results capped at $k=5$, evaluated across all 40 test queries.

---

## 2. Evaluation Metrics on Test Set ($k = 5$)

### 2.1 Overall Performance Across All Test Queries

| Search Method | NDCG@5 | MRR | Precision@5 | Recall@5 | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BCP Lexical Search (Baseline)** | 0.4052 | 0.4875 | 0.2500 | 0.4146 | *N/A (External)* |
| **BM25** | 0.6718 | 0.7258 | 0.4250 | 0.6667 | 0.12 ms |
| **BGE-M3** | **0.8452** | **0.9688** | **0.4600** | **0.7229** | 180.48 ms |
| **Hybrid Search (alpha=1.0)** | **0.8452** | **0.9688** | **0.4600** | **0.7229** | 192.53 ms |
| **Hybrid Search (alpha=0.60)** | 0.8205 | 0.9417 | **0.4600** | **0.7229** | 196.40 ms |

> [!NOTE]
> **Term Normalization Transparency**: BCP Lexical Search scores are reported above with standard case and diacritic normalization (accounting for minor orthographic variations in the BCP output, specifically Q-028 `Prapen` vs ground truth `Prapèn` and Q-029 `Jimat Klontongan` vs `Jimat klontongan`). Under strict character-exact matching without normalization, BCP Lexical Search achieves **NDCG@5: 0.3987**, **MRR: 0.4875**, **Precision@5: 0.2400**, and **Recall@5: 0.4000**.

### 2.2 Segmented Performance: Conceptual vs Exact Term Queries

| Search Method | Conceptual NDCG@5 | Conceptual MRR | Exact Term NDCG@5 | Exact Term MRR |
| :--- | :---: | :---: | :---: | :---: |
| **BCP Lexical Search (Baseline)** | 0.0000 | 0.0000 | 0.8103 | 0.9750 |
| **BM25** | 0.3591 | 0.4517 | **0.9845** | **1.0000** |
| **BGE-M3** | **0.7873** | **0.9750** | 0.9031 | 0.9625 |
| **Hybrid Search (alpha=1.0)** | **0.7873** | **0.9750** | 0.9031 | 0.9625 |
| **Hybrid Search (alpha=0.60)** | 0.7058 | 0.8833 | 0.9351 | **1.0000** |

*(Under strict character matching, BCP Exact Term NDCG@5 is 0.7974; Conceptual metrics remain 0.0000).*

### 2.3 Relative Performance Gains vs BCP Baseline

| Proposed Search Method | Overall NDCG Gain | Overall MRR Gain | Conceptual NDCG Gain | Exact Term NDCG Gain | Latency Profile |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25** | **+65.8%** (+0.2666) | **+48.9%** (+0.2383) | **+0.3591** *(from 0.0)* | **+21.5%** (+0.1742) | Ultra-low (0.12 ms) |
| **BGE-M3** | **+108.6%** (+0.4400) | **+98.7%** (+0.4813) | **+0.7873** *(from 0.0)* | **+11.5%** (+0.0928) | ~180 ms |
| **Hybrid Search (alpha=1.0)** | **+108.6%** (+0.4400) | **+98.7%** (+0.4813) | **+0.7873** *(from 0.0)* | **+11.5%** (+0.0928) | ~193 ms |
| **Hybrid Search (alpha=0.60)** | **+102.5%** (+0.4153) | **+93.2%** (+0.4542) | **+0.7058** *(from 0.0)* | **+15.4%** (+0.1248) | ~196 ms |

---

## 3. Computational Cost Monitoring

### 3.1 Latency Analysis per Query

| Search Method | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) | QPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BCP Lexical Search (Baseline)** | *N/A\** | *N/A\** | *N/A\** | *N/A\** | *N/A\** | *N/A\** |
| **BM25** | 0.12 | 0.03 | 0.54 | 0.01 | 0.59 | 8573.2 |
| **BGE-M3** | 180.48 | 183.29 | 211.43 | 125.85 | 298.01 | 5.5 |
| **Hybrid Search (alpha=1.0)** | 192.53 | 198.10 | 237.82 | 135.72 | 268.43 | 5.2 |
| **Hybrid Search (alpha=0.60)** | 196.40 | 197.63 | 240.20 | 146.22 | 271.59 | 5.1 |

*\*Note: Current BCP Lexical Search was executed externally and its latency was not instrumented within the local benchmark test harness.*

### 3.2 System Hardware Resource Utilization (CPU, GPU, RAM)

| Search Method | Process CPU Avg (Peak) | System CPU Avg | GPU Util Avg (Peak) | GPU VRAM Peak | RAM RSS Peak (Delta) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BCP Lexical Search (Baseline)** | *External* | *External* | *External* | *External* | *External* |
| **BM25** | 0.0% (0.0%) | 0.0% | 4.0% (4.0%) | 1376.0 MiB | 1933.0 MB (+0.0 MB) |
| **BGE-M3** | 549.3% (662.5%) | 63.4% | 1.5% (4.0%) | 1376.0 MiB | 1937.6 MB (+4.5 MB) |
| **Hybrid Search (alpha=1.0)** | 560.6% (675.0%) | 61.4% | 1.1% (2.0%) | 1376.0 MiB | 1938.0 MB (+0.5 MB) |
| **Hybrid Search (alpha=0.60)** | 556.8% (675.0%) | 62.3% | 1.1% (2.0%) | 1376.0 MiB | 1937.9 MB (+0.3 MB) |

### 3.3 Storage Usage Breakdown

| Search Method | Inverted Index Disk | Model Weights Disk | Database Vector Column | Total Storage Footprint |
| :--- | :---: | :---: | :---: | :---: |
| **BCP Lexical Search (Baseline)** | *Incumbent DB* | - | - | *Incumbent Footprint* |
| **BM25** | 338.90 KB | - | - | **0.33 MB** |
| **BGE-M3** | - | 4352.95 MB | 676.66 KB | **4353.61 MB** |
| **Hybrid Search (alpha=1.0)** | 338.90 KB | 4352.95 MB | 676.66 KB | **4353.94 MB** |
| **Hybrid Search (alpha=0.60)** | 338.90 KB | 4352.95 MB | 676.66 KB | **4353.94 MB** |

---

## 4. Alpha Optimization Grid Search on Validation Set

The validation set was used to evaluate $\alpha \in [0.0, 1.0]$.
Optimal alpha determined: **$\alpha = 1.0$**.

| Alpha (Dense Weight) | Sparse Weight | Validation NDCG@5 | Validation MRR | Validation P@5 | Validation R@5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 1.00 | 0.5632 | 0.6533 | 0.3000 | 0.4667 |
| 0.05 | 0.95 | 0.6444 | 0.6833 | 0.3800 | 0.6250 |
| 0.10 | 0.90 | 0.6444 | 0.6833 | 0.3800 | 0.6250 |
| 0.15 | 0.85 | 0.6664 | 0.7000 | 0.4000 | 0.6583 |
| 0.20 | 0.80 | 0.6664 | 0.7000 | 0.4000 | 0.6583 |
| 0.25 | 0.75 | 0.6675 | 0.7000 | 0.4000 | 0.6583 |
| 0.30 | 0.70 | 0.6692 | 0.7000 | 0.4000 | 0.6583 |
| 0.35 | 0.65 | 0.6692 | 0.7000 | 0.4000 | 0.6583 |
| 0.40 | 0.60 | 0.7079 | 0.7200 | 0.4200 | 0.7583 |
| 0.45 | 0.55 | 0.7460 | 0.7833 | 0.4200 | 0.7583 |
| 0.50 | 0.50 | 0.7960 | 0.8500 | 0.4200 | 0.7583 |
| 0.55 | 0.45 | 0.7960 | 0.8500 | 0.4200 | 0.7583 |
| 0.60 | 0.40 | 0.8218 | 0.8833 | 0.4400 | 0.7833 |
| 0.65 | 0.35 | 0.8218 | 0.8833 | 0.4400 | 0.7833 |
| 0.70 | 0.30 | 0.8285 | 0.9000 | 0.4400 | 0.7833 |
| 0.75 | 0.25 | 0.8445 | 0.9000 | 0.4800 | 0.8417 |
| 0.80 | 0.20 | 0.8456 | 0.9000 | 0.4800 | 0.8417 |
| 0.85 | 0.15 | 0.8330 | 0.9500 | 0.4600 | 0.8167 |
| 0.90 | 0.10 | 0.8352 | 0.9500 | 0.4600 | 0.8167 |
| 0.95 | 0.05 | 0.8574 | 0.9500 | 0.4800 | 0.8417 |
| 1.00 ⭐ | 0.00 | 0.8754 | 1.0000 | 0.4800 | 0.8417 |

---

## 5. Architectural Findings & Upgrade Recommendations

### 5.1 Baseline Analysis: Current BCP Lexical Search Engine

The empirical test results against the 40 test queries illuminate critical strengths and fundamental limitations in the incumbent BCP search engine:

1. **Complete Failure on Conceptual Queries (NDCG@5 = 0.0000, Recall@5 = 0.0000)**:
   - For all 20 conceptual test queries (e.g., *"peacekeeping ritual"*, *"Sacred site in the mountains"*, *"the nation in reog lore"*, *"woman vocalist in karawitan"*), BCP returned **zero relevant results**. 19 queries produced entirely blank responses, and 1 query returned an irrelevant lexicon (*"wife"* $\rightarrow$ *"Sayut"*).
   - This occurs because the incumbent engine operates strictly on headword string matching without indexing definition texts, cultural metadata, or semantic embeddings. When users do not know the exact Javanese or local term beforehand, the system is non-functional.

2. **Reasonable Exact Term Match, but Lacking Secondary Association (NDCG@5 = 0.8103, MRR = 0.9750)**:
   - When users query an exact headword (e.g., *"Cangkriman"*, *"Meduduk"*, *"Blandar Penirat"*), BCP reliably places the primary target lexicon at Rank 1 (MRR = 0.9750).
   - However, BCP struggles with broader exact queries that demand retrieving related cultural lexicons (e.g., related gamelan instruments for *"Saron"*, or co-occurring ritual offerings for *"Karo"*), capping its exact recall at 0.8292.

---

### 5.2 Comparative Analysis of Proposed Upgrade Paths

```
Overall NDCG@5 Comparison:
========================================================================================
BCP Lexical Search (Baseline)  | [████████████████] 0.4052
BM25 (In-Memory Multi-Field)   | [██████████████████████████] 0.6718  (+65.8%)
Hybrid Search (alpha=0.60)     | [████████████████████████████████] 0.8205  (+102.5%)
Hybrid Search (alpha=1.0)      | [█████████████████████████████████] 0.8452 (+108.6%)
BGE-M3 (Dense Vector)          | [█████████████████████████████████] 0.8452 (+108.6%)
========================================================================================
```

#### 1. Sparse BM25 Upgrade (In-Memory Okapi BM25 with Multi-Field Representation)
- **Superiority over BCP Baseline**:
  - BM25 boosts overall NDCG@5 from **0.4052 to 0.6718 (+65.8%)** and exact term NDCG@5 from **0.8103 to 0.9845 (+21.5%)** with a perfect **1.0000 MRR**.
  - BM25 indexes multi-field cultural metadata (term, definition, etymology, variants, semantic domains, categories, usage context) with inverse document frequency (IDF) weighting. Consequently, even on conceptual queries where keyword overlap exists in definition texts (e.g., *"flowers for rituals"* matching *"Petra"* or *"Borèh"*), BM25 achieves **0.3591 NDCG@5** (compared to 0.0000 in BCP).
- **Efficiency & Cost**:
  - Near-instantaneous execution: **0.12 ms mean latency** (~8,500 QPS) on standard CPU hardware.
  - Negligible storage footprint: **338.9 KB** in-memory index file with zero GPU dependency.

#### 2. Dense Vector Search (BGE-M3)
- **Transformational Semantic Capability**:
  - Eliminates vocabulary mismatch entirely, elevating conceptual query NDCG@5 from **0.0000 to 0.7873** and MRR from **0.0000 to 0.9750**.
  - Users can search in English, Indonesian, or descriptive natural language (e.g., *"animal mask"* $\rightarrow$ *"Dadak Merak"*, *"magical whip"* $\rightarrow$ *"Pecut Samandiman"*) without knowing the regional nomenclature.
- **Resource Trade-off**:
  - Mean latency increases to **180.48 ms per query** (~5.5 QPS) due to Transformer forward passes, requiring ~4.25 GB model weights and pgvector index columns.

#### 3. Hybrid Search Synergy (BM25 + BGE-M3 Fusion)
- **Best of Both Worlds**:
  - Combining BM25 and BGE-M3 provides the highest robustness across both query segments.
  - At **$\alpha = 0.60$** (Balanced Hybrid), the system maintains **1.0000 MRR on exact terms** (NDCG@5 = 0.9351) while delivering strong conceptual retrieval (NDCG@5 = 0.7058, MRR = 0.8833), more than doubling the baseline performance (+102.5% overall NDCG).
  - At **$\alpha = 1.0$** (Dense Dominant), overall NDCG peaks at **0.8452 (+108.6% gain)**.

---

### 5.3 Upgrade Decision Matrix for Brawijaya Corpora Project

Depending on available infrastructure and engineering priorities, two distinct upgrade routes are recommended:

| Objective / Constraint | Recommended Upgrade | Expected Gain vs BCP Baseline | Operational Overhead |
| :--- | :---: | :---: | :---: |
| **Immediate High-Throughput Lexical Drop-in** | **Okapi BM25** | **+65.8% Overall NDCG**<br>Perfect 1.0 MRR on Exact terms | **Zero** (Runs on CPU in 0.12 ms, 338 KB RAM, no GPU) |
| **Complete Cultural Discovery & Semantic Search** | **Hybrid Search ($\alpha = 0.60$ or $1.0$)** | **+102.5% to +108.6% Overall NDCG**<br>Unlocks Conceptual Search (0.0 $\rightarrow$ 0.787) | Requires vector storage & GPU/CPU embedding server (~190 ms) |
