from pipeline_rfc import pipeline_rfc
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import pandas as pd

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)

y = iris.target

rfc_model = pipeline_rfc()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

rfc_model.fit(X_train, y_train)

y_predict = rfc_model.predict(X_test)

print(classification_report(y_predict, y_test))

joblib.dump(rfc_model, "random_farest.joblib")