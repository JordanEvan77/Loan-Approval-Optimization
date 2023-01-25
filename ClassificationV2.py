import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import decimal
import seaborn as sns
import warnings
import shap
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import cross_val_score

low_memory = False
warnings.simplefilter(action='ignore', category=FutureWarning)

path = 'C:/Users/jorda/OneDrive/Desktop/PyCharm Community Edition 2021.2.2/5500 Capstone/Mid Term/Raw ' \
       'Data/'

## First Model Decision Tree! ##
data_final = pd.read_csv(path + 'data_final.csv')
data_final.dtypes


X = data_final[['LoanNr_ChkDgt', 'City', 'Zip', 'Bank', 'BankState', 'NAICS',
                     'ApprovalYear', 'ApprovalMonth', 'Term', 'NoEmp', 'NewExist', 'CreateJob',
                    'RetainedJob', 'FranchiseCode', 'UrbanRural', 'LowDoc', 'DisbursementGross',
                     'BalanceGross','SBA_Appv','ApprovalYear', 'protected']]


y = data_final['MIS_Status']

#ADASYN ADJUSTMENT! RE BALANCING DATA:
#https://www.kaggle.com/code/drscarlat/fraud-detection-under-oversampling-smote-adasyn

adasyn = ADASYN()
X_adjusted, y_adjusted = adasyn.fit_resample(X,y)
print('ready')
#
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
print('Accuracy', accuracy_score(y_test, pred1))
print('Precision', precision_score(y_test, pred1))
#r2 0.7901012137794616
#MSE 0.0524688965217362
#Accuracy 0.9475311034782637
#Precision 0.9475346540056756
# Wow, somewhat disapointing first try.

########################
# Grid Search, Decision Tree
########################
p = {'max_depth': [5, 10, 20, 30, 40, 50, 60],
         'min_samples_split': [2,3,4],
         'min_samples_leaf': [1,2]}


clf1_gs = DecisionTreeClassifier()
grid = GridSearchCV(estimator=clf1_gs, param_grid=p)
grid.fit(X_train, y_train)
print('ready')
print(grid.best_params_) # {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2}
best_model = grid.best_estimator_
best_model.fit(X_train,y_train)
pred_grid = best_model.predict(X_test)
print('r2', r2_score(y_test, pred_grid))
print('MSE', mean_squared_error(y_test, pred_grid))
print('Accuracy', accuracy_score(y_test, pred_grid))
print('Precision', precision_score(y_test, pred_grid))
#r2 0.8133888409028633
#MSE 0.04664763323681789
#Accuracy 0.9533523667631821
#Precision 0.9583054657280168
# Very good results!

########################
# Cross Validation
#########################

tree_scores = cross_val_score(best_model, X_train, y_train, scoring =
"neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-tree_scores)
print('Scores', tree_rmse_scores)
print('Mean', tree_rmse_scores.mean()) #
print('SD', tree_rmse_scores.std())
print('ready')
#Mean 0.21849455909871743 # not the best mean Squared error we hoped for.
#SD 0.001298390584541495




fig1 = plt.figure(figsize=(25,20))
tree.plot_tree(clf, feature_names=['LoanNr_ChkDgt', 'City', 'Zip', 'Bank', 'BankState', 'NAICS',
                'ApprovalDate_Num', 'Term', 'NoEmp', 'NewExist', 'CreateJob',
                'RetainedJob', 'UrbanRural', 'RevLineCr', 'LowDoc', 'DisbursementGross',
                'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv', 'ApprovalYear', 'MIS_Status'],
                   class_names=['P I F', 'CHGOFF'],
                   filled=True) # works!





####################
#Logistic Regression
####################
#May need to reduce number of variables observed?
ss = StandardScaler()
ss_X = ss.fit_transform(X_adjusted)

X_train_ss, X_test_ss, y_train, y_test = train_test_split(ss_X, y_adjusted, test_size= 0.2,
                                                    random_state=21)

log_reg = LogisticRegression(max_iter=1000, solver='saga')
clf2 = log_reg.fit(X_train_ss, y_train)
pred2 = clf2.predict(X_test_ss)

print('r2', r2_score(y_test, pred2))
print('MSE', mean_squared_error(y_test, pred2))
#r2 0.9859809266470397
#MSE 0.0035047571924833264
# great

#######################
#Cross Validation
######################



########################
#Pulling out Coefficients
########################
print(clf2.coef_, clf2.intercept_)
#[[-4.84682119e-02 -2.26088753e-02  4.52668197e-02 -3.19589590e-01
#  -4.42034691e-01  8.82196683e-02 -2.35267984e-01  8.62131313e-03
#  -1.65814467e+00 -1.04246398e-01  7.29944426e-02  1.92850333e-01
#  -1.25638769e-01 -8.10121149e-02  1.00815320e-01 -4.50439375e-01
#  -1.11470776e+00 -2.12898083e-01  1.32019689e+02 -3.66560361e+00
#   3.82035987e+00 -2.35267984e-01]] [49.24870687]

#Now link these coefficients with their respective variables?




################
#K Nearest Neighbor
###############
knn = KNeighborsClassifier(
    n_neighbors = 3)
knn1 = knn.fit(X_train, y_train)
prediction2 = knn1.predict(X_test)


print('r2', r2_score(y_test, prediction2))
print('MSE', mean_squared_error(y_test, prediction2))
#r2 0.27410062829745363
#MSE 0.18147426580490872
#Not a very good model, I could increase the neighbor size, but might not be beneficial






#########################
#Random Forest Classifier
#########################
clf_rf = RandomForestClassifier(n_estimators = 100)
clf_rf.fit(X_train, y_train)

# performing predictions on the test dataset
predictionrf= clf_rf.predict(X_test)
print('ready')

print('r2', r2_score(y_test, predictionrf))
print('MSE', mean_squared_error(y_test, predictionrf))
print('Accuracy', accuracy_score(y_test, predictionrf))
print('Precision', precision_score(y_test, predictionrf))
#r2 0.8738395834588618
#MSE 0.03153661800445611
#Accuracy 0.9684633819955439
#Precision 0.9654707548064547
# Very reasonable results


#########################
# Grid Search, Random Forest
#########################

param_grid = [
    {'n_estimators': [3, 10, 30, 50, 100, 150], 'max_features':[2, 4, 6, 8, 10]},
    {'bootstrap': [False], 'n_estimators':[3, 10, 15], 'max_features': [2, 3, 4, 6]},
]

forest_class2 = RandomForestClassifier()
grid_search = GridSearchCV(forest_class2, param_grid, cv=5,
                           scoring="neg_mean_squared_error",
                           return_train_score=True)

grid_search.fit(X_train, y_train)
#the below takes a long time to run
grid_search.best_params_ # best parameters!

best_model = grid_search.best_estimator_
best_model.fit(X_train,y_train)
pred_gridrf = best_model.predict(X_test)
print('r2', r2_score(y_test, pred_gridrf))
print('MSE', mean_squared_error(y_test, pred_gridrf))
print('Accuracy', accuracy_score(y_test, pred_gridrf))
print('Precision', precision_score(y_test, pred_gridrf))

########################
# Cross Validation
#########################

forest_scores = cross_val_score(clf_rf, X_train, y_train, scoring =
"neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)
print('Scores', forest_rmse_scores)
print('Mean', forest_rmse_scores.mean()) #
print('SD', forest_rmse_scores.std())
print('ready')
# Scores [0.01494466 0.01435836 0.0155088  0.01435836 0.01405605 0.01465444
# 0.01708987 0.0163185  0.01522934 0.01494472]
# Mean 0.015146309560658983 # 1.5% of error, this seems very reasonable!
# SD 0.0008968484248125571

###########################
# SHAP VALUE EXTRACTION
############################


# Attempt on full set WORKING:
group = X_test.iloc[880]
shp = shap.Explainer(clf_rf)
shap_values = shp.shap_values(group)
print(shp.expected_value, shap_values)
shap.initjs()
shap_plot = shap.force_plot(shp.expected_value[1], shap_values[1], matplotlib=True,
                feature_names=X_test.columns[0:22],
                show=True, plot_cmap=['#77dd77', '#f99191'])



# FINALLY WORKS!!!
# will need to take off ChargeofPrin? Right?

##############################
#SUMMARY SIZE:
###############################
# This works, but is sloppy
explainer = shap.TreeExplainer(clf_rf)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, feature_names=X_test.columns)
plt.show()



#############Other attempt
clf_rf2 = RandomForestClassifier(n_estimators = 100)
clf_rf2.fit(X_train.iloc[:1000], y_train.iloc[:1000])

# performing predictions on the test dataset
predictionrf= clf_rf2.predict(X_test.iloc[:100])

print('r2', r2_score(y_test.iloc[:100], predictionrf))
print('MSE', mean_squared_error(y_test.iloc[:100], predictionrf))

explainer = shap.TreeExplainer(clf_rf2)
shap_values = explainer(X_test.iloc[:100])
shap.summary_plot(shap_values, feature_names=X_test.columns)
plt.show()
# above works

