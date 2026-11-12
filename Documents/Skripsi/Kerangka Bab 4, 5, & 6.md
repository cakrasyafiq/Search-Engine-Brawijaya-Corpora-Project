## BAB IV HASIL

> **Catatan Prinsip Bab IV (Sesuai Panduan Skripsi FILKOM UB Subbab 3.4.4.1):**  
> Bab ini berfungsi murni untuk melaporkan hasil pelaksanaan metode/pengujian dan menyajikan data faktual secara terurut dan logis[cite: 33]. Seluruh tabel dan gambar harus diberi nomor, judul, serta diacu secara eksplisit di dalam teks paragraf[cite: 33]. Hindari interpretasi teoritis mendalam atau spekulasi penyebab di bab ini[cite: 33].

### 4.1 Hasil Penentuan Bobot Parameter $\alpha$ Optimal (*Validation Set*)[cite: 27, 28]
* **4.1.1 Data Evaluasi Variasi Nilai Parameter $\alpha$**[cite: 27, 28]
  * **Tabel 4.1:** Perbandingan Skor Evaluasi *Validation Set* terhadap Variasi Parameter $\alpha$ ($0{,}0$ hingga $1{,}0$ dengan interval $0{,}1$ atau $0{,}05$)[cite: 27, 28].
    * Kolom: Nilai $\alpha$, Bobot BM25 ($\alpha$), Bobot BGE-M3 ($1-\alpha$), Precision@K, Recall@K, MRR, NDCG@K[cite: 18, 21, 27].
  * Paragraf pengantar yang mengacu ke Tabel 4.1 dan mendeskripsikan secara objektif rentang perolehan nilai pada metrik evaluasi[cite: 33].
* **4.1.2 Karakteristik Perubahan Skor Metrik terhadap Nilai $\alpha$**[cite: 28]
  * **Gambar 4.1:** Kurva Grafik Perubahan Metrik Evaluasi (Precision, Recall, MRR, NDCG) terhadap Kandidat Nilai $\alpha$[cite: 21, 28].
  * Paragraf deskriptif yang menunjukkan titik nilai $\alpha$ dengan performa tertinggi (*optimal score*) yang terpilih untuk digunakan pada pengujian *test set*[cite: 28, 29, 33].

### 4.2 Hasil Evaluasi Kualitas *Information Retrieval* (*Test Set*)[cite: 8, 27, 29]
* **4.2.1 Data Perbandingan Metrik Efektivitas Antarmetode**[cite: 8, 29]
  * **Tabel 4.2:** Hasil Pengujian Efektivitas *Information Retrieval* pada *Test Set*[cite: 8, 27, 29].
    * Baris: BM25 (Sparse Retrieval), BGE-M3 (Dense Retrieval), Hybrid Retrieval ($\alpha$ optimal)[cite: 8, 29].
    * Kolom: Precision@5, Precision@10, Recall@5, Recall@10, MRR, NDCG@5, NDCG@10[cite: 8, 19, 20, 21].
  * Paragraf deskriptif pembacaan data Tabel 4.2: memaparkan urutan performa antarmetode secara faktual dari skor tertinggi hingga terendah untuk setiap metrik[cite: 33].
* **4.2.2 Hasil Evaluasi Berdasarkan Kategori Karakteristik Kueri**[cite: 27, 29]
  * **Tabel 4.3:** Perbandingan Performa Metode Berdasarkan Kategori Kueri Uji[cite: 8, 27, 29].
    * Kategori A: Kueri leksikal eksak (istilah leksikon langsung)[cite: 13, 27].
    * Kategori B: Kueri deskriptif/konseptual dalam bahasa Indonesia (*vocabulary mismatch*)[cite: 15, 27].
    * Kategori C: Kueri konseptual berbahasa Inggris (*cross-lingual search*)[cite: 16, 26].
  * Narasi faktual mengenai dokumen-dokumen relevan yang berhasil atau gagal menempati peringkat teratas pada masing-masing kelompok kueri[cite: 33].

### 4.3 Hasil Pengukuran Beban Komputasi[cite: 8, 22, 29]
* **4.3.1 Durasi Respons Sistem (*Query Latency*)**[cite: 22, 29]
  * **Tabel 4.4:** Pengukuran Latensi Pemrosesan Kueri Antarmetode (dalam milidetik/ms)[cite: 22, 29].
    * Kolom: Metode, Rata-rata (*Mean Latency*), Median (P50), Persentil 95 (P95)[cite: 22].
  * **Gambar 4.2:** Diagram Batang Distribusi Waktu Respons Kueri per Metode[cite: 22].
  * Narasi pembacaan data waktu eksekusi penelusuran[cite: 33].
* **4.3.2 Durasi Pembangunan Indeks (*Indexing Time*)**[cite: 22]
  * **Tabel 4.5:** Perbandingan Waktu Pembangunan Struktur Indeks Dokumen Leksikon[cite: 22].
    * Waktu pembuatan *inverted index* BM25 vs waktu inferensi representasi vektor dokumen oleh model BGE-M3 dan pengindeksan pada *vector database*[cite: 22, 26, 27].
* **4.3.3 Kebutuhan Memori dan Ruang Penyimpanan (*Memory & Storage Footprint*)**[cite: 22]
  * **Tabel 4.6:** Konsumsi Ruang Penyimpanan Indeks (*Disk Footprint*) dan Memori (*RAM/VRAM Usage*)[cite: 22].
    * Kolom: Metode, Ukuran File Indeks di Media Penyimpanan (MB), Penggunaan Memori Saat *Idle* (MB), Penggunaan Memori Saat Pemrosesan Kueri (MB)[cite: 22].
  * Narasi ringkas yang merangkum data kebutuhan sumber daya komputasi[cite: 33].

---

## BAB V PEMBAHASAN

> **Catatan Prinsip Bab V (Sesuai Panduan Skripsi FILKOM UB Subbab 3.4.4.2):**  
> Bab ini berfungsi untuk menerjemahkan makna hasil penelitian, melakukan evaluasi kritis, menghubungkan temuan empiris dengan teori/literatur terkait di Bab II, melakukan analisis kesalahan (*error analysis*), serta menjawab seluruh rumusan masalah penelitian secara tuntas[cite: 8, 11, 33, 34].

### 5.1 Analisis Kritis Kualitas *Information Retrieval*[cite: 8, 33, 34]
* **5.1.1 Analisis Perilaku Metode terhadap Fenomena *Vocabulary Mismatch***[cite: 13, 15, 34]
  * Pembahasan alasan mendasar penurunan metrik BM25 pada kueri deskriptif dan parafrasa, dikaitkan dengan keterbatasan *exact term matching* dan representasi *sparse*[cite: 13, 15].
  * Pembahasan keunggulan BGE-M3 dalam memetakan kueri konseptual dan lintas bahasa ke dalam *dense semantic space*, dikaitkan dengan pemahaman semantik arsitektur *Transformer*[cite: 15, 16].
* **5.1.2 Efektivitas Fusi Skor pada Pendekatan *Hybrid Retrieval***[cite: 17, 18, 29, 34]
  * Pembahasan alasan fusi skor mampu menyeimbangkan kelemahan representasi leksikal dan semantik sehingga menghasilkan skor metrik yang lebih optimal[cite: 17, 29].
  * Evaluasi peran normalisasi Min-Max dalam menyetarakan rentang nilai BM25 dan BGE-M3 guna mengatasi kendala dominasi skala (*scale dominance*)[cite: 18].
  * Interpretasi makna dari perolehan bobot $\alpha$ optimal: porsi kontribusi leksikal versus semantik yang paling ideal untuk karakteristik korpus Brawijaya Corpora[cite: 19, 28].
* **5.1.3 Analisis Kasus Kegagalan Penelusuran (*Error Analysis*)**[cite: 33, 34]
  * Identifikasi dan pembedahan kasus-kasus kueri uji yang menghasilkan nilai evaluasi rendah pada *test set* (*failure analysis*)[cite: 33, 34].
  * Pembahasan faktor penyebab kegagalan: ambiguitas leksikon lokal Jawa Timur, pengaruh panjang atribut definisi leksikon terhadap representasi vektor, atau keterbatasan pemahaman model terhadap istilah dialek daerah tertentu[cite: 11, 14, 16].

### 5.2 Analisis Kritis Efisiensi dan Beban Komputasi[cite: 8, 22, 33, 34]
* **5.2.1 Analisis *Trade-off* Latensi Pemrosesan Kueri**[cite: 13, 16, 22]
  * Evaluasi perbandingan kecepatan antara pencarian leksikal berbasis *inverted index* dan proses inferensi kueri menggunakan model *deep learning* BGE-M3[cite: 13, 16, 22].
  * Pembahasan *overhead* komputasi tambahan pada *hybrid retrieval* yang harus menjalankan dua *pipeline* sekaligus serta proses normalisasi dan fusi skor secara *real-time*[cite: 17, 18, 22].
* **5.2.2 Analisis Kebutuhan Sumber Daya Infrastruktur**[cite: 7, 22]
  * Evaluasi kelayakan penggunaan memori (RAM/VRAM) dan kapasitas penyimpanan *vector database* pada lingkungan server repositori digital Brawijaya Corpora[cite: 7, 22].
  * Pembahasan skalabilitas infrastruktur komputasi jika jumlah data leksikon dan artefak budaya bertambah di masa mendatang[cite: 7, 11, 22].

### 5.3 Rekomendasi Pemilihan Pendekatan *Information Retrieval* untuk Brawijaya Corpora[cite: 8, 33, 34]
*(Bagian sintesis evaluasi kritis untuk menjawab Rumusan Masalah ke-3)*[cite: 8]
* **5.3.1 Evaluasi Keseimbangan Efektivitas dan Efisiensi Komputasi**[cite: 8, 22]
  * Matriks komparasi *trade-off*: menimbang apakah margin peningkatan akurasi (NDCG/MRR) sebanding dengan biaya komputasi tambahan (*latency* & kebutuhan memori)[cite: 20, 21, 22].
* **5.3.2 Penentuan Pendekatan Terbaik Berdasarkan Karakteristik Pengguna dan Sistem**[cite: 7, 8, 13]
  * Rekomendasi metode definitif dengan mempertimbangkan dua persona pengguna Brawijaya Corpora: akademisi/peneliti budaya (kueri leksikal eksak) versus masyarakat umum/wisatawan (kueri deskriptif atau lintas bahasa)[cite: 7, 13].
  * Pertimbangan integrasi teknis terhadap arsitektur backend repositori digital Brawijaya Corpora yang sedang berjalan[cite: 7, 13].

---

## PENYESUAIAN PADA BAB LAIN TERKAIT STRUKTUR INI

* **Bab 3 (Metodologi Penelitian - Subbab 3.7):**  
  Pastikan rincian spesifikasi perangkat keras (CPU, RAM, GPU, SSD) dan perangkat lunak (OS, Python, library PyTorch/BGE-M3, jenis Vector Database) ditulis tuntas pada metodologi eksperimen agar Bab 4 bisa langsung memaparkan data kuantitatif[cite: 16, 22, 26, 33].
* **Bab 6 (Penutup - Sesuai Subbab 3.4.5 Panduan Skripsi FILKOM UB):**[cite: 36]
  * **6.1 Kesimpulan:** Pernyataan ringkas, tegas, dan berbasis data yang menjawab 3 rumusan masalah secara berurutan (kualitas perbandingan retrieval, beban komputasi, dan metode rekomendasi terpilih)[cite: 8, 29, 36].
  * **6.2 Saran:** Rekomendasi teknis untuk penelitian lanjutan (misalnya eksplorasi *Cross-Encoder reranking*, metode fusi adaptif/RRF, atau penambahan dataset dialek bahasa daerah Jawa Timur)[cite: 29, 36].