## 5.1 Analisis Kritis Kualitas *Information Retrieval*[cite: 8, 33, 34]

### 5.1.1 Analisis Perilaku Metode terhadap Fenomena *Vocabulary Mismatch*[cite: 13, 15, 34]

* Pembahasan alasan mendasar penurunan metrik BM25 pada kueri deskriptif dan parafrasa, dikaitkan dengan keterbatasan *exact term matching* dan representasi *sparse*[cite: 13, 15].
* Pembahasan keunggulan BGE-M3 dalam memetakan kueri konseptual dan lintas bahasa ke dalam *dense semantic space*, dikaitkan dengan pemahaman semantik arsitektur *Transformer*[cite: 15, 16].

### 5.1.2 Efektivitas Fusi Skor pada Pendekatan *Hybrid Retrieval*[cite: 17, 18, 29, 34]

* Pembahasan alasan fusi skor mampu menyeimbangkan kelemahan representasi leksikal dan semantik sehingga menghasilkan skor metrik yang lebih optimal[cite: 17, 29].
* Evaluasi peran normalisasi Min-Max dalam menyetarakan rentang nilai BM25 dan BGE-M3 guna mengatasi kendala dominasi skala (*scale dominance*)[cite: 18].
* Interpretasi makna dari perolehan bobot $\alpha$ optimal: porsi kontribusi leksikal versus semantik yang paling ideal untuk karakteristik korpus Brawijaya Corpora[cite: 19, 28].

### 5.1.3 Analisis Kasus Kegagalan Penelusuran (*Error Analysis*)[cite: 33, 34]

* Identifikasi dan pembedahan kasus-kasus kueri uji yang menghasilkan nilai evaluasi rendah pada *test set* (*failure analysis*)[cite: 33, 34].
* Pembahasan faktor penyebab kegagalan: ambiguitas leksikon lokal Jawa Timur, pengaruh panjang atribut definisi leksikon terhadap representasi vektor, atau keterbatasan pemahaman model terhadap istilah dialek daerah tertentu[cite: 11, 14, 16].
