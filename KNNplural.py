import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Veri setini dosyadan yükleyin
iris = pd.read_csv('C:/Users/xxxx/xxxxx/IRIS/Iris.csv')

# Özellikleri ve hedef değişkeni ayırın
X = iris.drop(columns=['Id', 'Species']).astype(float)
y = iris['Species']

# data görüntele
print(iris.head())

# classlarý görüntüle
print(iris['Species'].unique())

# data degiskenlerini ayarla
print(iris.describe(include='all'))
print(iris.info())

# Veriyi eğitim ve test setlerine bölün
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# k-NN sınıflandırıcısını oluşturun
knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')
knn.fit(X_train, y_train)

# Test verisi üzerinde tahminler yapın
y_pred = knn.predict(X_test)

# Başarım metriklerini hesaplayın
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')

accuracy_percentage = accuracy * 100
recall_percentage = recall * 100
precision_percentage = precision * 100

print(f"Doğruluk: {accuracy_percentage:.2f}%")
print(f"Duyarlılık: {recall_percentage:.2f}%")
print(f"Özgüllük: {precision_percentage:.2f}%")

