# BAB 1 PENDAHULUAN

## 1.1 Latar Belakang

Brawijaya Corpora Project merupakan inisiatif yang bertujuan untuk mendokumentasikan dan melestarikan pengetahuan serta ekspresi budaya khas masyarakat Jawa Timur dalam bentuk korpus digital. Dalam korpus ini, Brawijaya Corpora menerapkan pendekatan multimodal dengan mengintegrasikan berbagai jenis data seperti teks, gambar, audio, video, dan model 3D. Data tersebut kemudian dikelompokkan berdasarkan berbagai domain budaya untuk membentuk korpus pengetahuan budaya yang terstruktur dan mudah diakses (Yusron & Priyambadha, 2026).

Dalam sebuah sistem korpus yang menyimpan data dalam jumlah besar dan beragam, keberadaan *search engine* yang efektif menjadi komponen penting untuk memastikan informasi dapat ditemukan dan dimanfaatkan oleh pengguna. Sistem pencarian berfungsi sebagai jembatan yang menghubungkan data yang tersimpan dengan kebutuhan informasi pengguna. Dengan demikian, kualitas *information retrieval* memiliki pengaruh langsung terhadap pengalaman pengguna dalam menemukan leksikon, artefak, maupun informasi budaya yang relevan dengan kebutuhannya. Oleh karena itu, penyimpanan data yang terstruktur perlu didukung oleh mekanisme pencarian yang akurat dan mampu memahami kebutuhan pengguna.

Saat ini, pencarian pada sistem korpus umumnya masih menggunakan pendekatan *keyword-based search* yang mengandalkan kecocokan kata antara kueri pengguna dan data yang tersimpan. Pendekatan ini memiliki keterbatasan ketika pengguna menggunakan sinonim, parafrasa, variasi dialek, atau istilah yang berbeda untuk menggambarkan konsep yang sama. Permasalahan tersebut dapat menyebabkan informasi yang sebenarnya relevan tidak ditemukan karena tidak memiliki kecocokan kata secara langsung, sehingga menurunkan kualitas dan relevansi hasil pencarian (Hussan, 2020).

Salah satu pendekatan yang dapat mengatasi keterbatasan tersebut adalah *semantic search*. Berbeda dengan pencarian berbasis kata kunci, *semantic search* menggunakan model *deep learning* untuk merepresentasikan kueri dan dokumen berdasarkan makna serta konteksnya dalam bentuk vektor (Wang et al., 2022). Dengan pendekatan ini, sistem dapat memahami dan mencocokan konteks antara kueri dan data di database. Karena itu, metode ini berpotensi meningkatkan relevansi hasil pencarian, khususnya pada data budaya dan linguistik yang memiliki variasi istilah.

Penelitian ini bertujuan untuk melakukan analisis komparatif terhadap beberapa pendekatan *information retrieval*, yaitu *keyword based search*, *semantic search*, dan *hybrid search* yang menggabungkan kedua pendekatan tersebut. Setiap metode akan dievaluasi menggunakan *test query dataset* untuk mengidentifikasi pendekatan yang menghasilkan performa pencarian terbaik. Hasil penelitian ini akan digunakan sebagai dasar untuk mengimplementasikan solusi *search engine* yang paling efektif guna meningkatkan kemampuan *lexicon retrieval* pada sistem Brawijaya Corpora.

## 1.2 Rumusan Masalah

1. Bagaimana perbandingan kualitas *information retrieval* informasi antara pendekatan *keyword-based search*, *semantic search*, dan *hybrid search* pada dataset Brawijaya Corpora?
2. Bagaimana perbandingan beban komputasi antara metode *keyword-based search*, *semantic search*, dan *hybrid search* dalam memproses kueri pencarian terhadap dataset Brawijaya Corpora?
3. Pendekatan *information retrieval* manakah yang paling efektif untuk diimplementasikan pada sistem Brawijaya Corpora?

## 1.3 Tujuan

1. Menganalisis dan membandingkan kualitas *information retrieval* antara pendekatan *keyword-based search*, *semantic search*, dan *hybrid search* pada dataset Brawijaya Corpora.
2. Menganalisis dan membandingkan efisiensi komputasi dari pendekatan *keyword-based search*, *semantic search*, dan *hybrid search* dalam memproses kueri pencarian.
3. Menentukan pendekatan *information retrieval* yang paling efektif berdasarkan hasil evaluasi kualitas pencarian dan efisiensi komputasi untuk diimplementasikan pada sistem Brawijaya Corpora.

## 1.4 Manfaat

1. Bagi Brawijaya Corpora Project, penelitian ini dapat memberikan rekomendasi pendekatan *information retrieval* yang sesuai untuk meningkatkan kemampuan pencarian dan relevansi hasil *lexicon retrieval*.
2. Bagi pengembangan sistem pencarian, hasil penelitian dapat menjadi referensi dalam memilih pendekatan *keyword-based*, *semantic*, atau *hybrid retrieval* berdasarkan kualitas hasil pencarian dan efisiensi komputasinya.
3. Bagi penelitian selanjutnya, penelitian ini dapat menjadi referensi untuk penelitian mengenai penerapan dan evaluasi metode *information retrieval*, khususnya pada data korpus budaya dan linguistik.

## 1.5 Batasan Masalah

Untuk menjaga penelitian tetap terarah sesuai dengan rumusan masalah yang telah ditetapkan, batasan masalah dalam penelitian ini adalah sebagai berikut:

1. Penelitian difokuskan pada pencarian data leksikon yang tersedia pada Brawijaya Corpora Project.
2. Pendekatan *information retrieval* yang dibandingkan terdiri atas *keyword-based search* menggunakan algoritma BM25, *semantic search* menggunakan model BGE-M3, dan *hybrid search* yang menggabungkan keduanya.
3. Evaluasi kualitas hasil pencarian dilakukan menggunakan dataset *test query* yang telah dilengkapi dengan *relevance judgment* sebagai acuan relevansi hasil pencarian.
4. Metrik evaluasi kualitas *information retrieval* yang digunakan meliputi Precision@K, Recall@K, MRR@K, dan NDCG@K.
5. Evaluasi efisiensi difokuskan pada waktu pemrosesan kueri dan penggunaan sumber daya komputasi yang dibutuhkan oleh masing-masing pendekatan.
6. Penelitian tidak membahas pengembangan fitur pencarian lain di luar ketiga pendekatan *information retrieval* yang dibandingkan.
7. Hasil penelitian berupa rekomendasi pendekatan *information retrieval* terbaik berdasarkan hasil evaluasi dan menjadi dasar untuk implementasi pada sistem Brawijaya Corpora.

## 1.6 Sistematika Pembahasan

Sistematika pembahasan dalam penelitian ini terdiri atas lima bab yang disusun untuk memberikan gambaran mengenai tahapan penelitian mulai dari identifikasi permasalahan hingga analisis hasil penelitian dan penyusunan kesimpulan. Sistematika pembahasan penelitian ini adalah sebagai berikut:

**Tabel 1.1 Sistematika Pembahasan Penelitian**

| BAB | PEMBAHASAN |
| :--- | :--- |
| **BAB I Pendahuluan** | Bab ini membahas dasar pelaksanaan penelitian yang meliputi latar belakang, rumusan masalah, tujuan, manfaat, batasan masalah, dan sistematika pembahasan. Latar belakang menjelaskan permasalahan pencarian pada Brawijaya Corpora serta alasan pemilihan pendekatan *keyword-based search*, *semantic search*, dan *hybrid search* sebagai objek penelitian. |
| **BAB II Landasan Kepustakaan** | Bab ini membahas teori dan konsep yang menjadi dasar penelitian. Pembahasan meliputi Brawijaya Corpora Project, *information retrieval*, *keyword-based search*, *vocabulary mismatch*, *dense retrieval*, *hybrid retrieval*, serta evaluasi *information retrieval*. Bab ini digunakan sebagai landasan dalam menentukan metode, membangun sistem *retrieval*, dan melakukan evaluasi terhadap hasil pencarian. |
| **BAB III Metodologi** | Bab ini menjelaskan tahapan dan metode yang digunakan dalam penelitian. Tahapan penelitian meliputi jenis penelitian, tahapan penelitian, studi literatur, pengumpulan dataset, *dataset preprocessing*, pembuatan *test set*, dan pengujian. Pengujian mencakup pencarian nilai parameter terbaik pada metode *hybrid retrieval* serta perbandingan performa *keyword-based search*, *semantic search*, dan *hybrid search*. Hasil pengujian kemudian dievaluasi dan dianalisis untuk menentukan pendekatan yang paling efektif. |
| **BAB IV Hasil** | Bab ini menyajikan hasil implementasi dan pengujian dari pendekatan *keyword-based search*, *semantic search*, dan *hybrid search*. Hasil pengujian dianalisis berdasarkan kualitas hasil pencarian dan efisiensi komputasi. Bab ini juga membahas perbandingan performa ketiga pendekatan serta menentukan pendekatan yang paling sesuai untuk diterapkan pada sistem Brawijaya Corpora. |
| **BAB V Pembahasan** | Bab V ini menyajikan interpretasi makna hasil penelitian, analisis kritis komparatif, evaluasi mendalam, serta hubungan temuan empiris dengan literatur dan landasan teori yang telah dipaparkan pada Bab II. |
| **BAB VI Penutup** | Bab VI ini merupakan bagian penutup dari laporan skripsi yang menyajikan kesimpulan menyeluruh dari hasil penelitian dan pembahasan yang telah dilakukan, serta saran strategis bagi pengembangan penelitian dan implementasi sistem BCP kedepannya. |