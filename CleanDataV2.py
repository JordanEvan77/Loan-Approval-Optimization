import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import decimal
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action='ignore', category=FutureWarning)


low_memory = False

path = 'C:/Users/jorda/OneDrive/Desktop/PyCharm Community Edition 2021.2.2/5500 Capstone/Mid Term/Raw ' \
       'Data/'

raw_data_large= pd.read_csv(path + 'SBAnational.csv')

raw_data_large.describe()


# imports and read ins run as normal

raw_data_large.columns
# Index(['LoanNr_ChkDgt', 'Name', 'City', 'State', 'Zip', 'Bank', 'BankState',
#      'NAICS', 'ApprovalDate', 'ApprovalFY', 'Term', 'NoEmp', 'NewExist',
#     'CreateJob', 'RetainedJob', 'FranchiseCode', 'UrbanRural', 'RevLineCr',
#   'LowDoc', 'ChgOffDate', 'DisbursementDate', 'DisbursementGross',
#  'BalanceGross', 'MIS_Status', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv'],



##INITIAL CHANGES:
raw_data_large['ApprovalDate_Num'] = pd.to_datetime(raw_data_large['ApprovalDate']).dt.date
raw_data_large['ApprovalYear'] = pd.to_datetime(raw_data_large['ApprovalDate_Num']).dt.year
raw_data_large['DisbursementDate'] = pd.to_datetime(raw_data_large['DisbursementDate']).dt.date
raw_data_large['ChgOffDate'] = pd.to_datetime(raw_data_large['ChgOffDate']).dt.date
raw_data_large['NoEmp_num'] = raw_data_large['NoEmp'].values





##################################################
##### DATA MODIFICATION BEGINS #####
##################################################

### Missing data
raw_data_large.isna().sum()
# Name                     14
# City                     30
# State                    14
# Bank                   1559
# BankState              1566
# NewExist                136
# RevLineCr              4528
# LowDoc                 2582
# ChgOffDate           736465 # just needs to be filled in as not having one!
# DisbursementDate       2368
# MIS_Status             1997

########################
# impute or remove NAs
########################
raw_data_large['ChgOffDate'] = raw_data_large['ChgOffDate'].fillna(0)
clean_v1 = raw_data_large
# RevLineCr has a bunch of 'T' data points? Does that mean true? or something else? (should be y/n)
# RevLineCr has a bunch of 'T' data points? Does that mean true? or something else? (should be y/n)

# drop empty disbursement dates and MIS_status and LowDoc
clean_v2 = clean_v1.copy()
# clean_v2.dropna(subset=['BankState', 'MIS_Status', 'LowDoc']) # not necessary?
clean_v2 = clean_v2.dropna()
clean_v2.isna().sum()
# looks good!

# strip off dollar none of these are working lol
# will need to do this for 'BalanceGross', 'ChgOffPrinGr', 'GrAppv' and 'SBA_Appv'
print(clean_v2.dtypes)
clean_v2['BalanceGross'] = (clean_v2['BalanceGross'].str.strip('$'))
clean_v2['BalanceGross'] = clean_v2['BalanceGross'].str.replace(',', '').astype(float).astype(int)
clean_v2['BalanceGross'] = pd.to_numeric(clean_v2['BalanceGross'])  # finally works!

clean_v2['ChgOffPrinGr'] = (clean_v2['ChgOffPrinGr'].str.strip('$'))
clean_v2['ChgOffPrinGr'] = clean_v2['ChgOffPrinGr'].str.replace(',', '').astype(float).astype(int)
clean_v2['ChgOffPrinGr'] = pd.to_numeric(clean_v2['ChgOffPrinGr'])

clean_v2['GrAppv'] = (clean_v2['GrAppv'].str.strip('$'))
clean_v2['GrAppv'] = clean_v2['GrAppv'].str.replace(',', '').astype(float).astype(int)
clean_v2['GrAppv'] = pd.to_numeric(clean_v2['GrAppv'])

clean_v2['SBA_Appv'] = (clean_v2['SBA_Appv'].str.strip('$'))
clean_v2['SBA_Appv'] = clean_v2['SBA_Appv'].str.replace(',', '').astype(float).astype(int)
clean_v2['SBA_Appv'] = pd.to_numeric(clean_v2['SBA_Appv'])

clean_v2['DisbursementGross'] = (clean_v2['DisbursementGross'].str.strip('$'))
clean_v2['DisbursementGross'] = clean_v2['DisbursementGross'].str.replace(',', '').astype(float).astype(int)
clean_v2['DisbursementGross'] = pd.to_numeric(clean_v2['DisbursementGross'])


# SAVE FOR VISUALIZATION:
clean_v2.to_csv(path+'data_v2.csv')

# CHECK FOR OUTLIERS:

plt.boxplot(clean_v2[['BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']])
plt.xticks([1, 2, 3, 4], ['Gross Balance', 'Charge off Amount', 'Gross Amount Approved', 'SBA Gauranteed Amount'])
plt.show() # works well, heavy skew!


print(max(clean_v2['NoEmp_num']))
print(max(clean_v2['Term']))
plt.boxplot(clean_v2[['DisbursementGross', 'Term', 'NoEmp_num']])
plt.xticks([1, 2, 3], ['Disbursement Amount', 'Term of Loan', 'Number of Employees'])
plt.show() # disbursement amount and emp num are heavily skewed, Term is fine


## DATA NORMALIZATION (FOR SKEW): MIN MAX
# create a scaler object
scaler = MinMaxScaler()
# fit and transform the data
clean_v2= clean_v2.reset_index(drop=True)
clean_v3 = clean_v2.copy()

num_cols = clean_v2.select_dtypes(include=np.number).columns.tolist()
num_cols.remove('ApprovalYear') # removing non scale items

clean_v3[num_cols] = pd.DataFrame(scaler.fit_transform(clean_v3[num_cols]), columns=clean_v3[
    num_cols].columns)
clean_v3.isna().sum() # this was the issue
print(clean_v3.head(3))
#worked!




#Look at data imbalance: USE ADASIGN or Oversample or Undersample?
#https://www.kaggle.com/code/residentmario/oversampling-with-smote-and-adasyn/notebook

print(clean_v3['MIS_Status'].value_counts())
#P I F     730199
#CHGOFF    156041
# we will want to use adasyn when we are training our model!



# label encode:
labelencoder = LabelEncoder()

all_cols = clean_v3.dtypes

clean_v4 = clean_v3.copy()

# is there a better way to do this?
clean_v4['ApprovalDate_Num'] = labelencoder.fit_transform(clean_v3['ApprovalDate_Num'])
clean_v4['Bank'] = labelencoder.fit_transform(clean_v3['Bank'])
clean_v4['BankState'] = labelencoder.fit_transform(clean_v3['BankState'])
clean_v4['ChgOffDate'] = labelencoder.fit_transform(clean_v3['ChgOffDate'])  # a 0 or a date,
# necessary? since CHGOff Corresponds?
clean_v4['City'] = labelencoder.fit_transform(clean_v3['City'])
clean_v4['LowDoc'] = labelencoder.fit_transform(clean_v3['LowDoc'])
clean_v4['MIS_Status'] = labelencoder.fit_transform(clean_v3['MIS_Status'])
clean_v4['RevLineCr'] = labelencoder.fit_transform(clean_v3['RevLineCr'])
# all others work perfect!

# select final desired columns:
# Dont Need: DisbursementDate, State, Name (very miscelanious)

clean_v5 = clean_v4[['LoanNr_ChkDgt', 'City', 'Zip', 'Bank', 'BankState', 'NAICS',
                     'ApprovalDate_Num', 'Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob',
                     'FranchiseCode', 'UrbanRural', 'RevLineCr', 'LowDoc', 'DisbursementGross',
                     'BalanceGross',
                     'MIS_Status', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv', 'ApprovalYear']]
# took out 'ChgOffDate',  for now
clean_v5.to_csv(path+'data_final.csv')


# Check for Correlation:
sns.heatmap(clean_v5.corr())

### Key Features
# Count of CHGOFF per state, on the top 10 states
clean_v2['MIS_Status_num'] = labelencoder.fit_transform(clean_v2['MIS_Status'])

chg_off_df = clean_v2[clean_v2["MIS_Status_num"] == 0]  # works

states_group = pd.DataFrame(
    chg_off_df.groupby(["BankState"], as_index=False)["MIS_Status"].count())

# String change:
states_group["BankState"] = states_group["BankState"].astype('string')

top_states = states_group.sort_values("MIS_Status", ascending=False).groupby(
    ["BankState"]).head(2)  # why isnt this restricting rows?

top_states = top_states.head(10)

sns.barplot(x="BankState", y="MIS_Status", data=top_states)
# very cool! Should we look at Per Capita as well?


# rolling average visual of charge off over time
# needs changes
avg_viz = clean_v2
avg_viz['MIS_Dummy'] = 1 if avg_viz["MIS_Status_num"] == 0 else 0  # not working
avg_viz['rolling 7'] = avg_viz[avg_viz["MIS_Status_num"] == 0].count().rolling(7).mean()
plt.figure(figsize=(15, 6))
sns.lineplot(x='ApprovalYear', y='rolling 7', data=avg_viz)
plt.show()





###############
#THOUGHTS:
##############

#Handle Noisy data through binning, and fitting to bins means.
#Chi Square analysis: Finding redundant attributes
#Feature creation?






# DONT FORGET ASSIGNED QUESTIONS!
# 1. Identify at least 3 indicators or predictors of potential risk from the variables provided.
# Please review the variables provided and determine which variables would be valuable to predict
# the MIS_Status class. Provide a rationale for your decision as well as one quantitative method for
# confirming this. Please answer with at least 1 paragraph. (15 points)

# 2. Identify any preprocessing and data cleaning that might need to be done to this data
# beforehand. You do not have a copy of the data so this is a projection of possible preprocessing
# or data cleaning that might be helpful. Please answer with at least 1 paragraph. (15 points)

# 3. There are class imbalances, especially in the class we are trying to predict. Discuss the
# issues surrounding having class imbalances. Now discuss potential solutions to this class
# imbalance problem. Please answer with at least 2 paragraphs. (20 points)

# 4. At this point, the valuable features have been determined, the preprocessing and data
# cleaning completed and the class imbalances are handled. Now, we must choose a selection of 3
# models to test and and get results from. Identify these three models and explain in detail the
# process of using them as well as a brief discussion of the mathematical theory of each model you
# chose. Explain why these models were chosen. Consider issues around overfitting and underfitting
# as well as any hyperparameters that might need to be tuned. You are not expected to have selected
# a perfect set of models and hyperparameters but rather to show critical thinking in your decisions
# Please answer with a least 3 paragraphs. (40 points)

# 5. Finally, discuss the metrics that you would use for evaluation. Explain your rationale behind
# using those metrics. Please answer with at least 1 paragraph. (20 points)
