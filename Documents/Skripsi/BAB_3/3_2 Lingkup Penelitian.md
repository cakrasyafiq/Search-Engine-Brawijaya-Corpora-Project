## 3.2 Lingkup Penelitian

Lingkup dan batasan penelitian ini ditetapkan agar proses penelitian tetap terarah dan fokus pada tujuan utama yang telah dirumuskan. Batasan-batasan penelitian ini mencakup aspek-aspek berikut:

1. **Fokus pada Studi Komparatif, Bukan Implementasi Sistem Produksi (*Deployment*)**  
   Penelitian ini berfokus murni pada kajian komparatif (*comparative benchmarking*) untuk menganalisis performa efektivitas dan efisiensi komputasi dari metode BM25, BGE-M3, dan *hybrid retrieval*. Penelitian ini tidak mencakup implementasi langsung ke dalam sistem produksi (*live production*) repositori digital Brawijaya Corpora Project (BCP), perancangan antarmuka pengguna akhir (*frontend application / user interface*), maupun integrasi penuh ke dalam arsitektur layanan *backend* operasional BCP saat ini. Hasil dari penelitian ini diposisikan sebagai kajian empiris dan rekomendasi ilmiah bagi pengembang sistem Brawijaya Corpora dalam menentukan arah pengembangan arsitektur mesin pencari di masa mendatang.

2. **Karakteristik Korpus dan Data Penelitian**  
   Data yang digunakan dalam penelitian ini dibatasi pada korpus tekstual leksikon budaya khas Jawa Timur yang dihimpun dari repositori Brawijaya Corpora Project, yang terdiri dari 118 entri leksikon. Atribut teks yang digunakan mencakup istilah (*term*), definisi (*definition*), contoh penggunaan (*example*), dan konteks budaya atau etimologi (*cultural context / etymology*). Aset multimodal lain yang tersedia pada platform BCP, seperti rekaman suara, video pertunjukan, citra/foto artefak, dan model objek 3D, tidak dilibatkan dalam proses ekstraksi representasi maupun pengujian temu balik ini.

3. **Metode dan Algoritma yang Diuji**  
   Metode yang dikomparasikan dibatasi pada algoritma BM25 (BM25Okapi) sebagai representasi *sparse lexical retrieval*, model BGE-M3 (*pre-trained* BAAI/bge-m3 berdimensi 1024) sebagai representasi *dense retrieval*, serta penggabungan keduanya (*hybrid retrieval*) berbasis kombinasi linier terbobot (*weighted score fusion*) dengan fungsi normalisasi Min-Max. Penelitian tidak melakukan pelatihan ulang atau penyesuaian bobot lanjutan (*fine-tuning*) pada arsitektur model BGE-M3 (*zero-shot inference*).

4. **Penyusunan Dataset Evaluasi**  
   Dataset evaluasi disusun khusus untuk merepresentasikan skenario pencarian korpus budaya, yang mencakup variasi kueri leksikal eksak, kueri parafrasa deskriptif dalam bahasa Indonesia (*vocabulary mismatch*), dan kueri konseptual berbahasa Inggris (*cross-lingual*). Kumpulan data uji dibagi menjadi *validation set* (untuk proses penentuan parameter bobot fusi $\alpha$ optimal) dan *test set* (untuk evaluasi performa akhir komparatif).

5. **Lingkungan Pengujian Komputasi**  
   Pengukuran latensi dan beban komputasi dilakukan secara terstandarisasi pada lingkungan server uji eksperimen terisolasi (*test server environment*), bukan berdasarkan lalu lintas permintaan pengguna riil (*live concurrent traffic*) pada server BCP.
