from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
dataset['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
label_encoder = preprocessing.LabelEncoder()
dataset["species"]= label_encoder.fit_transform(dataset["species"]) # Encoding Output Variable
# Separating the Input and Output Columns
#print(dataset.columns)
X = dataset[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
y = dataset["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7)
gnb = GaussianNB()
clf = gnb.fit(X_train, y_train)
print("The accuracy on test set is {0:.2f}".format(clf.score(X_test, y_test)))
