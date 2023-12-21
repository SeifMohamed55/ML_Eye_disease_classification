import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
import matplotlib.pyplot as plt


dataset = pd.read_csv("Student_Performance.csv")
pd.set_option('display.max_columns', None)
# np.set_printoptions(threshold=np.inf)

# print(dataset.info())

X = dataset[["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours",
             "Sample Question Papers Practiced"]].copy()
y = dataset[["Performance Index"]]


encoder = LabelEncoder()
# dummies = pd.get_dummies(X["Extracurricular Activities"], prefix='Extracurricular Activities')
# dummies = dummies.astype(int)

X["Extracurricular Activities"] = encoder.fit_transform(X["Extracurricular Activities"])

# X.drop("Extracurricular Activities", axis=1, inplace=True)


#X = pd.concat([X, dummies], axis=1)

X = np.array(X.values)
y = np.array(y).flatten()

print(X)

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
#
# scaler = StandardScaler()
#
# # X_scaled_train = scaler.fit_transform(X_train)
# # X_scaled_test = scaler.transform(X_test)

# 0.9771272519212564 scaled

regressor = KNeighborsRegressor() # cross_val_score, KFold, GridSearchCV

# regressor.fit(X_scaled_train, y_train)
# print(regressor.score(X_scaled_test, y_test))

regressor.fit(X_train, y_train)


print(regressor.score(X_test, y_test))

# 0.9839703761874775

# 0.9889832909573145 linear regression
