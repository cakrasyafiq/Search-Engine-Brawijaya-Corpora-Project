## 2.3 Keyword-Based Search
*Keyword-based search* atau pencarian berbasis kata kunci merupakan pendekatan penelusuran informasi konvensional yang beroperasi atas prinsip pencocokan sintaksis (*syntactic matching*) antara token kata dalam kueri pengguna dengan teks dokumen yang tersimpan dalam repositori (Hussan, 2020). Pendekatan ini mengandalkan struktur *inverted index* dan representasi renggang (*sparse representation*) untuk memetakan frekuensi serta lokasi kemunculan setiap istilah di dalam korpus. Ketika pengguna memasukkan kueri, mesin pencari mengevaluasi kecocokan istilah untuk mengidentifikasi dan mengembalikan dokumen-dokumen yang memuat kata-kata tersebut.

Keunggulan mendasar dari *keyword-based search* terletak pada kesederhanaan implementasi serta efisiensi komputasinya yang sangat tinggi dalam memproses pencarian pada repositori teks berskala besar (Hussan, 2020). Sistem ini sangat presisi dan efisien apabila pengguna mencari entitas atau leksikon budaya tertentu menggunakan istilah yang eksak. Namun demikian, Hussan (2020) menegaskan bahwa kelemahan utama dari mesin pencari berbasis kata kunci adalah ketiadaan pemahaman terhadap makna intrinsik istilah maupun keterkaitan konseptual antarkata. Sistem ini rentan terhadap fenomena kebahasaan *vocabulary mismatch*, sehingga kerap memicu kegagalan penemuan dokumen relevan yang tidak memiliki kecocokan kata secara harfiah dengan kueri pengguna.

### 2.3.1 Algoritma BM25 (Best Matching 25)
Algoritma BM25 (*Best Matching 25*) atau Okapi BM25 merupakan algoritma yang berakar pada prinsip pemeringkatan probabilitas (*Probability Ranking Principle*) (Robertson & Zaragoza, 2009). BM25 memanfaatkan model probabilistik *2-Poisson* dan konsep *eliteness* untuk mengestimasi probabilitas relevansi sebuah dokumen terhadap kueri berdasarkan frekuensi kemunculan istilah (*term frequency*), kelangkaan istilah di seluruh korpus (*inverse document frequency*), serta normalisasi panjang dokumen.

Secara matematis, fungsi penilaian BM25 untuk menghitung skor relevansi dokumen $D$ terhadap kueri $Q$ yang terdiri dari sejumlah kata kunci $q_i$ dirumuskan sebagai berikut (Robertson & Zaragoza, 2009):

$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{tf(q_i, D) \cdot (k_1 + 1)}{tf(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

di mana:
- $tf(q_i, D)$ menyatakan frekuensi kemunculan kata kunci $q_i$ di dalam dokumen $D$.
- $|D|$ menyatakan panjang dokumen $D$ (jumlah total kata), sedangkan $\text{avgdl}$ (*average document length*) adalah rata-rata panjang dokumen pada keseluruhan koleksi korpus.
- $\text{IDF}(q_i)$ merupakan bobot *Inverse Document Frequency* yang mengukur tingkat spesifisitas atau daya pembeda dari istilah $q_i$, dirumuskan sebagai:

$$\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$$

dengan $N$ menyatakan total jumlah dokumen dalam korpus dan $n(q_i)$ adalah jumlah dokumen yang memuat kata kunci $q_i$. Kata-kata umum yang muncul di banyak dokumen akan memperoleh bobot IDF mendekati nol, sedangkan kata-kata yang langka dan spesifik akan memperoleh bobot diskriminasi yang tinggi.
