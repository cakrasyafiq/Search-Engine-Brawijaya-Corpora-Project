## 4.2 Hasil Evaluasi Kualitas *Information Retrieval* (*Test Set*)

Evaluasi kualitas temu balik informasi dilaksanakan pada *test set* independen yang terdiri atas **40 kueri uji unik** (20 kueri *conceptual* dan 20 kueri *exact term*, total 128 pasangan relevansi) pada kedalaman peringkat $k = 5$. Pengujian mencakup *BCP Lexical Search* (*baseline* eksternal), *Okapi BM25*, *BGE-M3*, serta *Hybrid Retrieval* pada $\alpha = 1{,}00$ dan konfigurasi pembanding $\alpha = 0{,}60$.

### 4.2.1 Data Perbandingan Metrik Efektivitas Antarmetode

Rekapitulasi skor efektivitas penelusuran keseluruhan (*overall*) pada *test set* ($k = 5$) disajikan pada Tabel 4.2. Skor *BCP Lexical Search* pada tabel tersebut dihitung dengan normalisasi kapitalisasi dan diakritik standar (`Prapen`/`Prapèn` serta `Jimat Klontongan`/`Jimat klontongan`); tanpa normalisasi karakter tersebut, *baseline* memperoleh Precision@5 sebesar 0,2400, Recall@5 sebesar 0,4000, MRR sebesar 0,4875, dan NDCG@5 sebesar 0,3987.

Tabel 4.2 Hasil Pengujian Efektivitas *Information Retrieval* pada *Test Set* ($k = 5$)

| Metode Penelusuran | Precision@5 | Recall@5 | MRR | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: |
| *BCP Lexical Search* (*Baseline*) | 0,2500 | 0,4146 | 0,4875 | 0,4052 |
| BM25 (*Sparse Retrieval*) | 0,4250 | 0,6667 | 0,7258 | 0,6718 |
| BGE-M3 (*Dense Retrieval*) | **0,4600** | **0,7229** | **0,9688** | **0,8452** |
| *Hybrid Retrieval* ($\alpha = 1{,}00$) | **0,4600** | **0,7229** | **0,9688** | **0,8452** |
| *Hybrid Retrieval* ($\alpha = 0{,}60$) | **0,4600** | **0,7229** | 0,9417 | 0,8205 |

Berdasarkan data pada Tabel 4.2, **BGE-M3** dan ***Hybrid Retrieval* ($\alpha = 1{,}00$)** menempati peringkat tertinggi secara bersama pada metrik NDCG@5 dengan skor **0,8452** (+108,6% dari *baseline*) dan MRR sebesar **0,9688** (+98,7%). Posisi berikutnya diikuti oleh ***Hybrid Retrieval* ($\alpha = 0{,}60$)** dengan NDCG@5 sebesar **0,8205** (+102,5%) dan MRR sebesar **0,9417** (+93,2%), **BM25** dengan NDCG@5 sebesar **0,6718** (+65,8%) dan MRR sebesar **0,7258** (+48,9%), serta ***BCP Lexical Search*** pada urutan terendah dengan NDCG@5 sebesar **0,4052** dan MRR sebesar **0,4875**. Sementara itu, pada metrik Precision@5 dan Recall@5, **BGE-M3**, ***Hybrid Retrieval* ($\alpha = 1{,}00$)**, dan ***Hybrid Retrieval* ($\alpha = 0{,}60$)** memperoleh nilai tertinggi yang setara sebesar **0,4600** dan **0,7229**, disusul oleh **BM25** sebesar **0,4250** dan **0,6667**, serta ***BCP Lexical Search*** sebesar **0,2500** dan **0,4146**.

### 4.2.2 Hasil Evaluasi Berdasarkan Kategori Karakteristik Kueri

Hasil pengujian berdasarkan dua kategori kueri pada *test set*, yaitu **kueri *exact term*** (20 kueri istilah leksikon langsung) dan **kueri *conceptual*** (20 kueri deskriptif berbahasa Inggris), disajikan pada Tabel 4.3. Pada pencocokan karakter ketat tanpa normalisasi kapitalisasi dan diakritik, skor NDCG@5 *BCP Lexical Search* pada segmen *exact term* tercatat sebesar 0,7974, sedangkan seluruh metrik segmen *conceptual* tetap bernilai 0,0000.

Tabel 4.3 Perbandingan Performa Metode Berdasarkan Kategori Kueri Uji ($k = 5$)

| Kategori Kueri | Metode Penelusuran | Precision@5 | Recall@5 | MRR | NDCG@5 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| ***Exact Term*** (20 kueri) | *BCP Lexical Search* (*Baseline*) | 0,5000 | 0,8292 | 0,9750 | 0,8103 |
| | BM25 (*Sparse Retrieval*) | **0,6000** | **0,9750** | **1,0000** | **0,9845** |
| | BGE-M3 (*Dense Retrieval*) | 0,5200 | 0,8375 | 0,9625 | 0,9031 |
| | *Hybrid Retrieval* ($\alpha = 1{,}00$) | 0,5200 | 0,8375 | 0,9625 | 0,9031 |
| | *Hybrid Retrieval* ($\alpha = 0{,}60$) | 0,5300 | 0,8542 | **1,0000** | 0,9351 |
| ***Conceptual*** (20 kueri) | *BCP Lexical Search* (*Baseline*) | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| | BM25 (*Sparse Retrieval*) | 0,2500 | 0,3583 | 0,4517 | 0,3591 |
| | BGE-M3 (*Dense Retrieval*) | **0,4000** | **0,6083** | **0,9750** | **0,7873** |
| | *Hybrid Retrieval* ($\alpha = 1{,}00$) | **0,4000** | **0,6083** | **0,9750** | **0,7873** |
| | *Hybrid Retrieval* ($\alpha = 0{,}60$) | 0,3900 | 0,5917 | 0,8833 | 0,7058 |

Berdasarkan data pada Tabel 4.3, pada kelompok **kueri *exact term***, **BM25** mencatat performa tertinggi (NDCG@5 = **0,9845**; MRR = **1,0000**; Precision@5 = **0,6000**; Recall@5 = **0,9750**) dengan menempatkan istilah target pada peringkat ke-1 di seluruh 20 kueri serta mencapai NDCG@5 = 1,0000 pada 17 kueri. Posisi berikutnya ditempati oleh ***Hybrid Retrieval* ($\alpha = 0{,}60$)** dengan NDCG@5 = **0,9351** dan MRR = **1,0000**, diikuti oleh **BGE-M3** dan ***Hybrid Retrieval* ($\alpha = 1{,}00$)** yang masing-masing memperoleh NDCG@5 = **0,9031** dan MRR = **0,9625** (19 dari 20 kueri menempati peringkat ke-1), serta ***BCP Lexical Search*** dengan NDCG@5 = **0,8103** dan MRR = **0,9750** (19 kueri menempati peringkat ke-1).

Sebaliknya, pada kelompok **kueri *conceptual***, **BGE-M3** dan ***Hybrid Retrieval* ($\alpha = 1{,}00$)** meraih skor tertinggi secara bersama (NDCG@5 = **0,7873**; MRR = **0,9750**; Precision@5 = **0,4000**; Recall@5 = **0,6083**) dengan keberhasilan mengembalikan dokumen relevan pada seluruh 20 kueri dan menempatkan 19 kueri pada peringkat ke-1. ***Hybrid Retrieval* ($\alpha = 0{,}60$)** menyusul dengan NDCG@5 = **0,7058** dan MRR = **0,8833** (17 kueri di peringkat ke-1 dan 1 kueri gagal mengembalikan dokumen relevan), diikuti oleh **BM25** dengan NDCG@5 = **0,3591** dan MRR = **0,4517** (7 kueri gagal mengembalikan dokumen relevan), sedangkan ***BCP Lexical Search*** memperoleh **0,0000** pada seluruh metrik karena 19 kueri menghasilkan respons kosong dan 1 kueri mengembalikan hasil tidak relevan.
