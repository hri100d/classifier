import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv(r'C:\Users\dimov\OneDrive\Desktop\SOZ_Project\framingham.csv')
dataset = dataset.dropna()

dataset = pd.get_dummies(dataset, columns = ['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes'])

standardScaler = StandardScaler()
columns_to_scale = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

y = dataset['TenYearCHD']
X = dataset.drop(['TenYearCHD'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

rf_scores = []
estimators = [10, 100, 200, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(rf_classifier.score(X_test, y_test))

colors = rainbow(np.linspace(0, 1, len(estimators)))
plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)
for i in range(len(estimators)):
    plt.text(i, rf_scores[i], rf_scores[i])
plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])
plt.xlabel('Number of estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different number of estimators')

print("The score for Random Forest Classifier is {}% with {} estimators.".format(rf_scores[0]*100, 10))
print("The score for Random Forest Classifier is {}% with {} estimators.".format(rf_scores[1]*100, 100))
print("The score for Random Forest Classifier is {}% with {} estimators.".format(rf_scores[2]*100, 200))
print("The score for Random Forest Classifier is {}% with {} estimators.".format(rf_scores[3]*100, 500))
print("The score for Random Forest Classifier is {}% with {} estimators.".format(rf_scores[4]*100, 1000))

# plt.show()