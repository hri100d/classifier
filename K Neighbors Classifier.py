import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv(r'C:\Users\dimov\OneDrive\Desktop\SOZ_Project\framingham.csv')
dataset = dataset.dropna()

dataset = pd.get_dummies(dataset, columns = ['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes'])

standardScaler = StandardScaler()
columns_to_scale = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

y = dataset['TenYearCHD']
X = dataset.drop(['TenYearCHD'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))

fig, ax = plt.subplots()
ax.plot([k for k in range(1, 21)], knn_scores, color='red')

for i, score in enumerate(knn_scores, start=1):
    ax.text(i, score, f'{score:.4f}', ha='center', va='bottom')

ax.set_xticks([i for i in range(1, 21)])
ax.set_xlabel('Number of Neighbors (K)')
ax.set_ylabel('Scores')
ax.set_title('K Neighbors Classifier scores for different K values')

print("The score for K Neighbors Classifier is {}% with {} nieghbors.".format(knn_scores[13]*100, 14))

#plt.show()