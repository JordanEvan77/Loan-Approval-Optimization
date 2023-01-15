import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import decimal
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error, confusion_matrix, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN

low_memory = False
warnings.simplefilter(action='ignore', category=FutureWarning)

path = 'C:/Users/jorda/OneDrive/Desktop/PyCharm Community Edition 2021.2.2/5500 Capstone/Mid Term/Raw ' \
       'Data/'

## First Model Decision Tree! ##
data_final = pd.read_csv(path + 'data_final.csv')
data_final.dtypes


X = data_final[['LoanNr_ChkDgt', 'City', 'Zip', 'Bank', 'BankState', 'NAICS',
                'ApprovalDate_Num', 'Term', 'NoEmp', 'NewExist', 'CreateJob',
                'RetainedJob', 'UrbanRural', 'RevLineCr', 'LowDoc', 'DisbursementGross',
                'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv', 'ApprovalYear']]


y = data_final['MIS_Status']

#ADASYN ADJUSTMENT! RE BALANCING DATA:
#https://www.kaggle.com/code/drscarlat/fraud-detection-under-oversampling-smote-adasyn

adasyn = ADASYN()
X_adjusted, y_adjusted = adasyn.fit_resample(X,y)
print('ready')

X_train, X_test, y_train, y_test = train_test_split(X_adjusted, y_adjusted, test_size= 0.2, random_state=
21)

###################
#DECISION TREE
##################
clf = DecisionTreeClassifier()
clf1 = clf.fit(X_train, y_train)
pred1 = clf1.predict(X_test)

print('r2', r2_score(y_test, pred1))
print('MSE', mean_squared_error(y_test, pred1))
#r2 0.9713480040616262
#MSE 0.007162806995846603
# Wow, very reasonable first attempt!!!





################
#K Nearest Neighbor
###############
knn = KNeighborsClassifier(
    n_neighbors = 3)
knn1 = knn.fit(X_train, y_train)
prediction2 = knn1.predict(X_test)


print('r2', r2_score(y_test, prediction2))
print('MSE', mean_squared_error(y_test, prediction2))
#r2 0.20553580758503665
#MSE 0.19861072462871926
#Not a very good model, I could increase the neighbor size, but might not be beneficial


#########################
#Random Forest Classifier
#########################
clf_rf = RandomForestClassifier(n_estimators = 100)
clf_rf.fit(X_train, y_train)

# performing predictions on the test dataset
predictionrf= clf_rf.predict(X_test)

print('r2', r2_score(y_test, predictionrf))
print('MSE', mean_squared_error(y_test, predictionrf))
#r2 0.9856533890840949
#MSE 0.0035865565964814643
#  really really good results!!


# should I do a grid search on this model?