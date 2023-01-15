import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
low_memory=False
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

path = 'C:/Users/jorda/OneDrive/Desktop/PyCharm Community Edition 2021.2.2/5500 Capstone/Mid Term/Raw ' \
       'Data/'

#import V2 of data and final version of data
clean_v2 = pd.read_csv(path+'data_v2.csv')



# HISTOGRAMS
clean_v2['MIS_Status'].hist(bins=50)
plt.show()

clean_v2['Term'].hist(bins=50)
plt.show()  # majority of loans are below 100 months, but large groups too at 250 and 300


clean_v2['ApprovalYear'].hist(bins=50)
plt.show()


plt.plot(clean_v2['NoEmp_num'].values)
plt.show()


# Check for Correlation:
sns.heatmap(clean_v2.corr())


###BAR PLOTS WITH CHARGE OFF Categorized:
#https://www.geeksforgeeks.org/create-a-stacked-bar-plot-in-matplotlib/
#CHG OFF On states


states_group = clean_v2.groupby(['State', 'MIS_Status']).size().unstack(level=1)
states_group=states_group.apply(lambda x: x*100/sum(x), axis=1)
states_group.plot(kind = 'bar')
plt.title("Percentage Comparison")
#specific states with more risk

### CHG OFF on Binned Company Size
binned_comp = clean_v2
bins = [0, 10, 20, 30, 40, 50, 100, 200, 300, 500, 800, 1000, 1200]

binned_comp['binned Emp No']=pd.cut(binned_comp['NoEmp'], bins)
size_group = binned_comp.groupby(['binned Emp No', 'MIS_Status']).size().unstack(level=1)
size_group = size_group.apply(lambda x: x*100/sum(x), axis=1)
size_group.plot(kind = 'bar')
plt.title("Percentage Comparison")
# smaller companies, more risk

### CHG OFF On Binned Loan Size
binned_loansz = clean_v2
bins = [0, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 300000, 500000, 1000000, 1200000,
        1500000, 1800000, 2000000, 2500000, 3000000]

binned_loansz['binned loansz']=pd.cut(binned_loansz['DisbursementGross'], bins)
size_ln = binned_loansz.groupby(['binned loansz', 'MIS_Status']).size().unstack(level=1)
size_ln = size_ln.apply(lambda x: x*100/sum(x), axis=1)
size_ln.plot(kind = 'bar')
plt.title("Percentage Comparison")
#smaller loans, more risk


### CHG OFF on Urbanrural
urban_group = clean_v2.groupby(['UrbanRural', 'MIS_Status']).size().unstack(level=1)
urban_group=urban_group.apply(lambda x: x*100/sum(x), axis=1)
urban_group.plot(kind = 'bar')
plt.title("Percentage Comparison")
#1 = Urban, 2 = Rural, 0 = undefined
# Urban has more charge off!


### CHG OFF on Newexist


### CHG OFF on Term



# TOP MOST RISKY STATES:
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



### OUTLIER VISUALS:
###
plt.boxplot(clean_v2[['BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']])
plt.xticks([1, 2, 3, 4], ['Gross Balance', 'Charge off Amount', 'Gross Amount Approved', 'SBA Gauranteed Amount'])
plt.show() # works well, heavy skew!


print(max(clean_v2['NoEmp_num']))
print(max(clean_v2['Term']))
plt.boxplot(clean_v2[['DisbursementGross', 'Term', 'NoEmp_num']])
plt.xticks([1, 2, 3], ['Disbursement Amount', 'Term of Loan', 'Number of Employees'])
plt.show() # disbursement amount and emp num are heavily skewed, Term is fine