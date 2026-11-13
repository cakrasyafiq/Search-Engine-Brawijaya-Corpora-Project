## 3.3 Lingkungan Penelitian

Lingkungan penelitian mencakup spesifikasi perangkat keras (*hardware*) dan perangkat lunak (*software stack*) yang digunakan untuk menjalankan proses pengindeksan, penelusuran kueri, serta pengukuran beban komputasi secara terisolasi. Rincian spesifikasi lingkungan uji yang digunakan dalam penelitian ini disajikan pada Tabel 3.2.

Tabel 3.2 Spesifikasi Perangkat Keras dan Perangkat Lunak Lingkungan Uji

| Kategori | Parameter / Komponen | Spesifikasi Lingkungan Uji |
| :--- | :--- | :--- |
| **Perangkat Keras (*Hardware*)** | Processor (CPU) | AMD Ryzen 5 3600X (Desktop) |
| | Memori (RAM) | 16 GB DDR4 2400 MHz |
| | GPU | NVIDIA GeForce GTX 1650 Super 4 GB |
| | Media Penyimpanan (*Disk*) | SSD 500 GB |
| **Perangkat Lunak (*Software Stack*)** | Sistem Operasi | Windows 11 |
| | Bahasa Pemrograman | Python 3.11 |
| | *Framework* API & Server | FastAPI, Uvicorn, Pydantic |
| | Basis Data & Kontainerisasi | PostgreSQL 16 dengan ekstensi `pgvector` (Docker: `pgvector/pgvector:pg16`), SQLAlchemy |
| | Model *Embedding* NLP | `BAAI/bge-m3` (*dense* 1024 dimensi dan *native learned sparse*) |
| | *Framework* & Pustaka *Deep Learning* | PyTorch, `sentence-transformers`, `transformers` |
| | Mesin Pencari Leksikal (BM25) | Implementasi kustom *in-memory inverted index* Okapi BM25 (`app/search/bm25_index.py`) |
| | Pengujian & Pemantauan Komputasi | `pytest`, `httpx`, `psutil` (pemantauan CPU/RAM dan latensi) |

Spesifikasi lingkungan uji pada Tabel 3.2 menjadi acuan terkontrol dalam mengukur kinerja temu balik dan beban komputasi masing-masing metode pada Bab IV dan Bab V.
