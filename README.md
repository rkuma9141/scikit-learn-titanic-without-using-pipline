# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 2. Load Titanic Dataset
data = pd.read_csv("titanic.csv")

# 3. Select Features & Target
X = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = data["Survived"]

# 4. Handle Missing Values
imputer = SimpleImputer(strategy="median")
X["Age"] = imputer.fit_transform(X[["Age"]])
X["Fare"] = imputer.fit_transform(X[["Fare"]])

# For categorical missing values
X["Embarked"] = X["Embarked"].fillna("S")

# 5. Encode Categorical Variables
encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded = encoder.fit_transform(X[["Sex", "Embarked"]])

# Combine numerical + encoded categorical
X_num = X[["Pclass", "Age", "SibSp", "Parch", "Fare"]].values
X_final = np.hstack([X_num, encoded])

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# 7. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 9. Prediction
y_pred = model.predict(X_test)

# 10. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
