# README - Decision Tree Classification: Citrus Dataset UTS Praktikum ML

**Nama:** Akhmad Ridlo Rifa'i 
**NIM:** 1227050013 
**Praktikum:** Pembelajaran Mesin - B

---

Dokumen ini menjelaskan tahapan pembuatan model klasifikasi untuk menentukan apakah suatu buah termasuk **jeruk (orange)** atau **anggur (grapefruit)** menggunakan **algoritma Decision Tree**.

---

## 1. Import Library yang Diperlukan

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
```

## 2. Load dan Eksplorasi Data

```python
data = pd.read_csv("citrus.csv")
print(data.head())
print(data["name"].value_counts())  # Memastikan distribusi kelas
```

## 3. Preprocessing Data

- Label encoding: mengubah kolom `name` menjadi numerik.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['label'] = le.fit_transform(data['name'])  # orange -> 1, grapefruit -> 0
```

- Pisahkan fitur dan label:

```python
X = data.drop(columns=['name', 'label'])
y = data['label']
```

## 4. Split Data Training dan Testing

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 5. Inisialisasi dan Latih Model Decision Tree

```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

## 6. Prediksi dan Evaluasi

```python
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

---

## Hasil & Evaluasi

- **Akurasi**: [nilai akurasi akan muncul dari script di atas]  
- **Confusion Matrix**: Menunjukkan jumlah klasifikasi benar dan salah untuk kedua kelas (jeruk dan anggur).  
- **Classification Report**: Meliputi precision, recall, dan f1-score.

---

## Catatan

- Algoritma Decision Tree dipilih karena interpretasinya mudah dan cocok untuk klasifikasi sederhana seperti ini.  
- Dataset cukup seimbang, sehingga tidak perlu penanganan imbalance data.  
- Model dapat diimprovisasi lebih lanjut dengan **cross-validation** atau **pruning**.

---

## Sumber Data

Dataset: `citrus.csv` - terdiri dari fitur-fitur statistik dari citra buah untuk klasifikasi dua kelas: `orange` dan `grapefruit`.

