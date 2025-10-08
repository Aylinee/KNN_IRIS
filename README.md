# KNN_IRIS
Iris dataset was used to perform classification using the k-NN (k-Nearest Neighbors) algorithm. The Iris dataset includes four features (sepal length, sepal width, petal length, and petal width) for three different iris flower species: Setosa, Versicolor, and Virginica
Veri seti, bir CSV dosyasından yüklenmiş ve özellikler ile hedef değişken ayrılmıştır. Iris veri seti, toplam 150 örnekten oluşmaktadır ve her bir örnek dört öznitelik ve bir hedef sınıftan oluşmaktadır. Veri ön işleme adımlarında, veri normalizasyonu ve eğitim-test ayrımı gerçekleştirilmiştir.
Python kod parçası
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

Project Summary: K-Nearest Neighbors (k-NN) Analysis on Iris Dataset
This project explores the impact of various data splitting parameters and model configurations on the performance of the k-Nearest Neighbors (k-NN) classification algorithm using the classic Iris dataset.

Data Preparation and Feature Engineering
The Iris dataset was loaded and prepared as follows:

Python

# Load the dataset
iris = pd.read_csv('C:/Users/xxxxx/xxxxx/IRIS/Iris.csv')

# Separate features and the target variable
X = iris.drop(columns=['Id', 'Species']).astype(float)
y = iris['Species']
Analysis Scenarios: Impact of test_size and random_state
We tested the model's robustness and generalization ability by using three different test sizes and two different random states to partition the data.

Program File	Test Size	Random State	Purpose
KNNplural.py	0.4 (40%)	42	Using a larger test set (40%).
K-NnwithWeighted.py	0.3 (30%)	42	Standard test size (30%).
irisK-NN.py	0.3 (30%)	2	Same test size (30%) but different data split using random_state=2.
KNNWithIRIS.py	0.2 (20%)	42	Using a smaller test set (20%).

E-Tablolar'a aktar
The data splitting code structure for each scenario:

Python

# Example for KNNplural.py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Example for irisK-NN.py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
k-NN Model Setup and Training
A k-NN classifier was instantiated with k=3 neighbors. This value was chosen because the Iris dataset has three classes (Setosa, Versicolor, Virginica), and k=3 provides a good balance: low k values increase sensitivity to noise, while high k values over-generalize the model.

Scenario 1: Uniform Weighting (Plurality)
The first program (KNNplural.py) used a uniform weighting scheme, where all neighbors have equal influence:

Python

# Create the k-NN classifier with uniform weights
knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')
knn.fit(X_train, y_train)
Scenario 2: Default k-NN
The other programs (irisK-NN.py, KNNWithIRIS.py, K-NnwithWeighted.py) used the default settings (which is also uniform weighting):

Python

# Create the k-NN classifier with default settings (uniform weights)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
Model Evaluation and Performance Metrics
The models were evaluated by making predictions on the test set and calculating standard classification metrics: Accuracy, Recall, and Precision.

Python

# Make predictions on the test data
y_pred = knn.predict(X_test)
We examined the effect of the averaging method in metric calculation:

Standard Evaluation (Macro Average)
Used in KNNplural.py, irisK-NN.py, and KNNWithIRIS.py. Macro averaging calculates metrics independently for each class and then takes the unweighted average, treating all classes equally.

Python

# Calculate performance metrics using Macro Average
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
Weighted Evaluation (Weighted Average)
Used in K-NnwithWeighted.py. Weighted averaging is used to account for class imbalance by weighting the average by the number of true instances for each class.

Python

# Calculate performance metrics using Weighted Average
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
Output Formatting
All results are presented as percentages:

Python

accuracy_percentage = accuracy * 100
recall_percentage = recall * 100
precision_percentage = precision * 100

print(f"Accuracy: {accuracy_percentage:.2f}%")
print(f"Recall: {recall_percentage:.2f}%")
print(f"Precision: {precision_percentage:.2f}%")


irisK-NN>>%100 



<img width="620" height="539" alt="image" src="https://github.com/user-attachments/assets/d458fcfb-fc63-4360-93dc-953b5bce8fe8" />



KNNplural.py>>%98

<img width="492" height="580" alt="image" src="https://github.com/user-attachments/assets/5590ed31-1365-4f6d-a6a6-6f2e2d17eaf3" />







KNNwithIRIS.py>>%100



<img width="502" height="508" alt="image" src="https://github.com/user-attachments/assets/a4fca680-bcba-4807-a943-c4aae56f2019" />




K-NJNwithWeighted>>%100 

<img width="625" height="565" alt="image" src="https://github.com/user-attachments/assets/9ae9967d-7002-4dd0-af47-166d481ca0b7" />


##Conclusion and Performance Summary
This study successfully implemented the k-Nearest Neighbors (k-NN) algorithm for the classification of the Iris dataset.

The model's performance was rigorously evaluated using key metrics: Accuracy, Recall, and Precision. Across multiple testing scenarios (different test sizes and random states), the k-NN algorithm demonstrated exceptionally high success rates, achieving 100% classification accuracy in several tests.

This confirms k-NN as a simple, yet highly effective non-parametric classification method, particularly well-suited for datasets like Iris where class separation is distinct.
