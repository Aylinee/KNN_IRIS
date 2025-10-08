# KNN_IRIS
Iris dataset was used to perform classification using the k-NN (k-Nearest Neighbors) algorithm. The Iris dataset includes four features (sepal length, sepal width, petal length, and petal width) for three different iris flower species: Setosa, Versicolor, and Virginica
Veri seti, bir CSV dosyasından yüklenmiş ve özellikler ile hedef değişken ayrılmıştır. Iris veri seti, toplam 150 örnekten oluşmaktadır ve her bir örnek dört öznitelik ve bir hedef sınıftan oluşmaktadır. Veri ön işleme adımlarında, veri normalizasyonu ve eğitim-test ayrımı gerçekleştirilmiştir.
Python kod parçası
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Veri setini dosyadan yükleyin
iris = pd.read_csv('C:/Users/AylinF/Desktop/IRIS/Iris.csv')

# Özellikleri ve hedef değişkeni ayırın
X = iris.drop(columns=['Id', 'Species']).astype(float)
y = iris['Species']

Veri işlemenin KNN’deki veri doğruluğunu test size’ın önemini bildiğimişz için 3 farklı test size kullanıldı.İlk program KNNplural.py’de 0.4 yani datanın %40’ı test olarak kullanıldı. Random state 42 olarak girildi. 

# Veriyi eğitim ve test setlerine bölün
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

K-NnwithWeighted programında %30 test datası olarak kullanılmıştır.

 # Veriyi eğitim ve test setlerine bölün
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

 irisK-NN.py programında test değeri %30 olarak belirlenmiş olup random state değeri 2 olarak seçilmiştir. 

# Veriyi eğitim ve test setlerine bölün
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

KNNWithIRIS.py programındaysa, %20 test kullanıldı ve Random state aynı seçildi. 

# Veriyi eğitim ve test setlerine bölün
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


k-NN Modeli Oluşturma
k-NN sınıflandırıcısı oluşturulmuş ve uygun k değeri (k=3) seçilmiştir. değerinin 3 olarak belirlenmesinin temel sebebi, Iris veri setinde bulunan üç sınıf (Setosa, Versicolor, Virginica) arasındaki ayrımın dengeli bir şekilde gerçekleştirilmesini sağlamasıdır. Düşük k değerleri algoritmanın gürbüzlük seviyesini azaltabilirken, yüksek k değerleri ise modelin fazla genelleşmesine neden olabilir. k=3, komşuluk yapısı için optimal bir dengenin sağlandığı değer olarak belirlenmiştir. Model, eğitim verisi ile eğitilmiştir.
İlk değerlendirme Plural olarak en yakın komşuluğun değerlendirilmesini sağlanmıştır. Bu sebeple kod içinde bu kısım şu şekilde belirtilmiştir. 

# k-NN sınıflandırıcısını oluşturun
knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')
knn.fit(X_train, y_train)
Diğer değerlendirmelerde pythonda normal K-NN algoritmasını görmek için aşağıdaki gibi kodlandı (irisK-NN.py KNNwithIRIS.py K-NnwithWeighted.py ):

# k-NN sınıflandırıcısını oluşturun
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
Modeli Değerlendirme
Model, test verisi üzerinde tahminler yapmış ve doğruluk, duyarlılık ve özgüllük metrikleri hesaplanmıştır. Bu metrikler, modelin performansını değerlendirmek için kullanılmıştır.
python
# Test verisi üzerinde tahminler yapın
y_pred = knn.predict(X_test)

Başarım metriklerinde K-NN’in sağladığı  avantajları değerlendirmek için 3 farklı durum değerlendirilmiştir. KNNwithWeighted.py dışındakiler aşağıdaki şekilde başarımları hesaplanmıştır.


# Başarım metriklerini hesaplayın
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')


Ağırlıklı hesaplamyı karşılaştırmak için hazırlanan  KNNwithWeighted.py program parçası şuşekildedir:

# Başarım metriklerini hesaplayın
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')


İstenen başarım sonuçlarının yüzdelik oplarak gösterilmesi için aşğıdaki gösterim yapılmıştır. 
accuracy_percentage = accuracy * 100
recall_percentage = recall * 100
precision_percentage = precision * 100

print(f"Doğruluk: {accuracy_percentage:.2f}%")
print(f"Duyarlılık: {recall_percentage:.2f}%")
print(f"Özgüllük: {precision_percentage:.2f}%")

Sonuçlar
<img width="940" height="822" alt="image" src="https://github.com/user-attachments/assets/d7d369dc-7178-4d73-823e-712179547228" />

irisK-NN>>%100 


<img width="940" height="1112" alt="image" src="https://github.com/user-attachments/assets/80e097ac-467a-4f94-a2db-a5d7bc1da2f9" />



KNNplural.py>>%98

<img width="940" height="944" alt="image" src="https://github.com/user-attachments/assets/5dc4c235-7cc0-4f68-ab6a-37bf8f7e3d86" />







KNNwithIRIS.py>>%100

<img width="940" height="944" alt="image" src="https://github.com/user-attachments/assets/32b93108-be73-42c9-83c5-252fdc266f59" />



K-NJNwithWeighted>>%100 

<img width="940" height="812" alt="image" src="https://github.com/user-attachments/assets/38be6a9e-f98c-4732-a27f-f6195ba8e8d5" />


Tartışma
Bu çalışmada, k-NN algoritması kullanılarak Iris veri seti üzerinde sınıflandırma işlemi gerçekleştirilmiştir. Modelin performansı, doğruluk, duyarlılık ve özgüllük metrikleri ile değerlendirilmiş ve yüksek başarı elde edilmiştir. k-NN algoritması, basit ve etkili bir sınıflandırma yöntemi olarak bu tür veri setlerinde %100 başarılı sonuçlar vermektedir. 

