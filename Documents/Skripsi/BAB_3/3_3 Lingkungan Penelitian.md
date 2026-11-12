## 3.3 Lingkungan Penelitian

Lingkungan penelitian mencakup infrastruktur komputasi perangkat keras (*hardware*) dan rangkaian perangkat lunak (*software stack*) yang digunakan untuk mengeksekusi seluruh tahapan eksperimen temu balik informasi. Lingkungan ini disiapkan secara khusus untuk mendukung kebutuhan komputasi intensif dalam pemrosesan data, kalkulasi indeks BM25, inferensi model representasi semantik BGE-M3, penelusuran ruang vektor, serta kalkulasi metrik evaluasi secara objektif.

Eksperimen pengujian dilakukan pada lingkungan server uji (*test server environment*) berbasis *cloud virtual environment* yang dilengkapi dengan akselerator grafis (GPU). Penggunaan GPU ditujukan untuk mempercepat inferensi model berbasis *deep learning* BGE-M3 dalam proses *document embedding* dan *query encoding*. Sementara itu, pengukuran beban komputasi dan performa *retrieval* dilakukan secara terisolasi guna menghindari interferensi proses komputasi lain.

Untuk memberikan gambaran yang komprehensif mengenai perbedaan antara lingkungan server uji yang digunakan dalam penelitian ini dengan infrastruktur server yang mengoperasikan sistem Brawijaya Corpora saat ini, Tabel 3.1 menyajikan perbandingan spesifikasi perangkat keras dan *tech stack* dari kedua lingkungan tersebut. Data spesifikasi server Brawijaya Corpora Project (BCP) saat ini disediakan dalam bentuk kolom isian agar dapat dilengkapi setelah informasi resmi infrastruktur BCP diperoleh.

Tabel 3.1 Komparasi Spesifikasi dan Tech Stack Server BCP Saat Ini dengan Lingkungan Server Uji

| Kategori | Parameter / Komponen | Server BCP Saat Ini | Lingkungan Server Uji (Eksperimen) | Keterangan / Fungsi |
| :--- | :--- | :--- | :--- | :--- |
| **Perangkat Keras (Hardware)** | Processor (CPU) | | Intel(R) Xeon(R) CPU @ 2.20GHz (2 vCPU) | Pemrosesan logika aplikasi, tokenisasi BM25, dan fusi skor |
| | Memori Utama (RAM) | | 12.67 GB – 15.6 GB RAM | Penyimpanan data korpus dan matriks indeks *in-memory* |
| | Akselerator Grafis (GPU) | | NVIDIA Tesla T4 (15.6 GB GDDR6 VRAM) | Akselerasi inferensi model *embedding* BGE-M3 (FP16) |
| | Media Penyimpanan (Disk) | | SSD Persistent Storage (~100 GB) | Penyimpanan dataset, bobot model, dan *log* pengujian |
| **Perangkat Lunak (Software Stack)** | Sistem Operasi | | Linux (Ubuntu 22.04 LTS x86_64) | Sistem operasi lingkungan server pengujian |
| | Bahasa Pemrograman | | Python 3.10+ | Bahasa utama untuk *pipeline* pengujian dan evaluasi |
| | Framework Deep Learning | | PyTorch 2.11.0 (CUDA 12.x) | Eksekusi komputasi tensor dan inferensi BGE-M3 |
| | Model Embedding NLP | | BAAI/bge-m3 | Model representasi semantik (*dense retrieval*) |
| | Library Information Retrieval | | `rank_bm25` (BM25Okapi) | Algoritma *sparse retrieval* berbasis kata kunci |
| | Library Representasi Semantik | | `FlagEmbedding`, `transformers` | Pustaka tokenisasi dan inferensi model BGE-M3 |
| | Basis Data & Vector Store | | PostgreSQL 15+ (`pgvector`) / PyTorch Tensor | Penyimpanan leksikon dan penelusuran vektor semantik |
| | Framework Layanan API | | FastAPI / Uvicorn | *Service backend* untuk pengujian latensi *endpoint* |
| | Library Analisis Data & Metrik | | `numpy`, `pandas`, `scikit-learn` | Kalkulasi fusi skor terbobot dan metrik evaluasi IR |

Perbedaan spesifikasi antara server BCP saat ini dan lingkungan server uji menjadi salah satu pertimbangan penting dalam analisis efisiensi komputasi pada Bab V, khususnya terkait rekomendasi arsitektur mesin pencari yang realistis dan dapat diterapkan secara optimal pada sistem Brawijaya Corpora.
