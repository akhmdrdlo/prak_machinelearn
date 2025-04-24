# README - Decision Tree Classification: Citrus Dataset UTS Prak ML

**Nama:** Akhmad Ridlo Rifa'i 
**NIM:** 1227050013  
**Praktikum:** Pembelajaran Mesin - B

---

Oke jadi... di file ini kita bakal ngebahas step by step gimana caranya kita bisa bikin model klasifikasi sederhana untuk nentuin buah ini tuh jeruk atau anggur ya. Gak usah ribet, kita pakai **Decision Tree** karena dia paling gampang dimengerti dan cocok buat beginner kayak kita.

---

## 1. Import Library

Langkah paling awal, tentu aja kita perlu load semua library yang kita butuh. Kita bakal pakai `pandas` buat data, `scikit-learn` buat model dan evaluasi.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
```

---

## 2. Baca dan Lihat-lihat Datanya

Oke, sekarang kita load file `.csv`-nya dan lihat dulu data kita isinya apa aja. Juga kita cek jumlah buah jeruk sama anggur ada berapa biar tau balance atau enggak.

```python
data = pd.read_csv("citrus.csv")
print(data.head())
print(data["name"].value_counts())  # Cek jumlah kelas jeruk vs anggur
```

---

## 3. Preprocessing (Bersihin dan Siapin Data)

Nah sekarang kita ubah label `name` yang tadinya teks ("orange", "grapefruit") jadi angka biar model ngerti.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['label'] = le.fit_transform(data['name'])  # orange = 1, grapefruit = 0 (otomatis)
```

Habis itu, kita pisahkan fitur (X) dan target (y).

```python
X = data.drop(columns=['name', 'label'])  # fitur
y = data['label']  # label yang mau diprediksi
```

---

## 4. Split Data (Latih dan Uji)

Supaya fair, kita pisah data jadi dua: 80% buat latihan, 20% buat testing.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 5. Buat dan Latih Model Decision Tree

Sekarang waktunya kita panggil Decision Tree dan latih dia pakai data training yang udah kita siapkan.

```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

---

## 6. Prediksi dan Lihat Hasilnya

Kita minta model buat nebak data test, dan kita bandingin hasilnya sama label aslinya.

```python
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

---

## Hasil & Evaluasi

Dari hasil evaluasi di atas, kita bisa lihat:

- **Akurasi**: Seberapa sering model nebak dengan benar.
- **Confusion Matrix**: Kotak 2x2 yang nunjukin prediksi vs kenyataan.
- **Classification Report**: Detail banget—ada precision, recall, sama f1-score buat masing-masing kelas.

---

## Sumber Data

Dataset yang dipakai: `citrus.csv` - berisi fitur statistik buah (mungkin dari gambar atau pengukuran lainnya) yang tujuannya buat bantu model klasifikasi antara `orange` dan `grapefruit`.

---

Kalau kamu mau eksperimen lebih lanjut, bisa juga coba tweaking parameter Decision Tree atau bandingin sama algoritma lain kayak Random Forest atau KNN. Tapi ini udah cukup banget buat dasar klasifikasi buah 😄

