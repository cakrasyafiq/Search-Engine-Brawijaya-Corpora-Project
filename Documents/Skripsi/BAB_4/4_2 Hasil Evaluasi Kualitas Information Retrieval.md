## 4.2 Hasil Evaluasi Kualitas *Information Retrieval* (*Test Set*)

Evaluasi kualitas temu balik informasi dilaksanakan pada *test set* independen yang terdiri atas **40 kueri uji unik** (20 kueri *conceptual* dan 20 kueri *exact term*, total 128 pasangan relevansi) pada kedalaman peringkat $k = 5$. Pengujian mencakup *BCP Lexical Search* (*baseline* eksternal), *Okapi BM25*, *BGE-M3*, serta *Hybrid Retrieval* pada $\alpha = 1{,}00$ dan konfigurasi pembanding $\alpha = 0{,}60$.

### 4.2.1 Data Perbandingan Metrik Efektivitas Antarmetode

Rekapitulasi skor efektivitas penelusuran keseluruhan (*overall*) pada *test set* ($k = 5$) disajikan pada Tabel 4.2.

Tabel 4.2 Hasil Pengujian Efektivitas *Information Retrieval* pada *Test Set* ($k = 5$)

| Metode Penelusuran | Precision@5 | Recall@5 | MRR | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: |
| *BCP Lexical Search* (*Baseline*) | 0,2500 | 0,4146 | 0,4875 | 0,4052 |
| BM25 (*Sparse Retrieval*) | 0,4250 | 0,6667 | 0,7258 | 0,6718 |
| BGE-M3 (*Dense Retrieval*) | **0,4600** | **0,7229** | **0,9688** | **0,8452** |
| *Hybrid Retrieval* ($\alpha = 1{,}00$) | **0,4600** | **0,7229** | **0,9688** | **0,8452** |
| *Hybrid Retrieval* ($\alpha = 0{,}60$) | **0,4600** | **0,7229** | 0,9417 | 0,8205 |

> **Catatan *Baseline*:** Skor *BCP Lexical Search* pada Tabel 4.2 menggunakan normalisasi kapitalisasi dan diakritik (`Prapen`/`Prapèn` dan `Jimat Klontongan`/`Jimat klontongan`). Tanpa normalisasi tersebut, skor *baseline* tercatat Precision@5 = 0,2400; Recall@5 = 0,4000; MRR = 0,4875; dan NDCG@5 = 0,3987.

Berdasarkan Tabel 4.2, urutan performa antarmetode dari tertinggi ke terendah adalah sebagai berikut:

1. **NDCG@5:** **BGE-M3** dan ***Hybrid Retrieval* ($\alpha = 1{,}00$)** mencatat skor tertinggi sebesar **0,8452** (+108,6% dari *baseline*), diikuti *Hybrid Retrieval* ($\alpha = 0{,}60$) sebesar **0,8205** (+102,5%), BM25 sebesar **0,6718** (+65,8%), dan *BCP Lexical Search* sebesar **0,4052**.
2. **MRR:** **BGE-M3** dan ***Hybrid Retrieval* ($\alpha = 1{,}00$)** meraih skor tertinggi sebesar **0,9688** (+98,7%), diikuti *Hybrid Retrieval* ($\alpha = 0{,}60$) sebesar **0,9417** (+93,2%), BM25 sebesar **0,7258** (+48,9%), dan *BCP Lexical Search* sebesar **0,4875**.
3. **Precision@5 dan Recall@5:** **BGE-M3**, ***Hybrid Retrieval* ($\alpha = 1{,}00$)**, dan ***Hybrid Retrieval* ($\alpha = 0{,}60$)** memperoleh nilai tertinggi yang setara (Precision@5 = **0,4600**; Recall@5 = **0,7229**), diikuti BM25 (0,4250; 0,6667) dan *BCP Lexical Search* (0,2500; 0,4146).

### 4.2.2 Hasil Evaluasi Berdasarkan Kategori Karakteristik Kueri

Hasil pengujian berdasarkan dua kategori kueri pada *test set*, yaitu **kueri *exact term*** (20 kueri istilah leksikon langsung) dan **kueri *conceptual*** (20 kueri deskriptif berbahasa Inggris), disajikan pada Tabel 4.3.

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

> **Catatan *Exact Term Baseline*:** Tanpa normalisasi karakter, NDCG@5 *BCP Lexical Search* pada segmen *exact term* adalah **0,7974**, sedangkan segmen *conceptual* tetap **0,0000**.

Berdasarkan Tabel 4.3, ringkasan pembacaan data untuk masing-masing kelompok kueri adalah sebagai berikut:

1. **Kueri *Exact Term* (20 kueri):** **BM25** mencatat performa tertinggi (NDCG@5 = **0,9845**; MRR = **1,0000**; Recall@5 = **0,9750**) dengan menempatkan istilah target pada peringkat ke-1 di seluruh 20 kueri dan mencapai NDCG@5 = 1,0000 pada 17 kueri. ***Hybrid Retrieval* ($\alpha = 0{,}60$)** menyusul dengan NDCG@5 = **0,9351** dan MRR = **1,0000**, diikuti oleh **BGE-M3** dan ***Hybrid Retrieval* ($\alpha = 1{,}00$)** (NDCG@5 = **0,9031**; MRR = **0,9625**, dengan 19 dari 20 kueri menempati peringkat ke-1). ***BCP Lexical Search*** memperoleh NDCG@5 = **0,8103** dan MRR = **0,9750** (19 kueri menempati peringkat ke-1).
2. **Kueri *Conceptual* (20 kueri):** **BGE-M3** dan ***Hybrid Retrieval* ($\alpha = 1{,}00$)** meraih skor tertinggi (NDCG@5 = **0,7873**; MRR = **0,9750**) dengan keberhasilan mengembalikan dokumen relevan pada seluruh 20 kueri (19 kueri di peringkat ke-1). ***Hybrid Retrieval* ($\alpha = 0{,}60$)** mencatat NDCG@5 = **0,7058** dan MRR = **0,8833** (17 kueri di peringkat ke-1, 1 kueri gagal mengembalikan dokumen relevan). **BM25** memperoleh NDCG@5 = **0,3591** dan MRR = **0,4517** dengan 7 kueri gagal mengembalikan dokumen relevan, sedangkan ***BCP Lexical Search*** memperoleh **0,0000** pada seluruh metrik (19 kueri menghasilkan respons kosong dan 1 kueri mengembalikan hasil tidak relevan).
