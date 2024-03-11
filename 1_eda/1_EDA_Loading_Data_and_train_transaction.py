#!/usr/bin/env python
# coding: utf-8

# # Credit-Card Fraud Detection
# ## Background and Objective
# 
# ---
# 
# 
# 
# ---
# 
# 
# **BACKGROUND**
# 
# Istitute of Elelctrical and Electronics Enngineers-Computational Intelligence society (IEEE-CIS) works across a variety of AI and machine learning areas, including deep neural networks, fuzzy systems, evolutionary computation, and swarm intelligence. They had partnered with the world’s leading payment service company, Vesta Corporation, seeking the best solutions for fraud prevention industry.
# 
# **OBJECTIVE**:
# 
# **The task is to  benchmark machine learning models on a challenging large-scale dataset. The model(s) will improve the efficacy of fraudulent transaction alerts for millions of people around the world, helping hundreds of thousands of businesses reduce their fraud loss and increase their revenue.**
# 
# 
# ## Dataset Description
# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# **Dataset Description**:
# 
# The dataset contains 590,540 card transactions, 20,663 of which are fraudulent (3.5%). Each transaction has **431 features** (400 numerical, 31 categorical), along with the relative **timestamp** and a label of whether it was fraudulent or legitimate. For anonymization purposes, the names of the identity features have been masked, along with the names of the extra features engineered by Vesta.
# 
# In this project, we evaluate the probability/anomaly score that an online transaction is fraudulent, as denoted by the binary target isFraud.
# 
# The data is broken into two files: **identity** and **transaction**, which are joined by TransactionID. Not all transactions have corresponding identity information.
# 
# **Categorical Features - Transaction**:
# - ProductCD
# - card1-card6
# - addr1
# - addr2
# - P_emaildomain
# - R_emaildomain
# - M1 - M9
# 
# **Categorical Features - Identity**:
# - DeviceType
# - DeviceInfo
# - id_12 - id_38
# 
# The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).
# 
# **Files**:
# - **train_transaction.csv, train_identity.csv**: the training set
# - **test_transaction.csv, test_identity.csv**: the test set (To predict the isFraud value for these observations)
# 
# As we can notice, the datasets we have been provided with contain 2 files, namely, a. transaction and b. identity for each train and test set.
# 
# 🔑 **URL to download the dataset**: [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
# 
# 🔑 **Additional Resources**: [Benchmark models-Papers with code](https://paperswithcode.com/dataset/ieee-cis-fraud-detection-1)
# 
# ## More Information about the Dataset
# 
# 
# ---
# 
# 
# **Transaction Table**
# 
# TransactionDT: Represents the time difference from a specified reference datetime (not an actual timestamp).
# TransactionAMT: Denotes the payment amount for each transaction in USD.
# ProductCD: Signifies the product code corresponding to each transaction.
# card1 - card6: Includes payment card details such as card type, category, issuing bank, country, etc.
# addr: Indicates the address associated with the transaction.
# dist: Represents the distance.
# P_ and (R__) emaildomain: Denotes the email domain of the purchaser and recipient respectively.
# C1-C14: Refers to counting values, for instance, the number of addresses linked with the payment card, with the actual meaning concealed.
# D1-D15: Represents time differences, such as the number of days between previous transactions.
# M1-M9: Indicates matches, such as matches between names on cards and addresses.
# Vxxx: Denotes Vesta engineered features which are rich in attributes such as ranking, counting, and other entity relations.
# 
#     **Categorical Features:**
# 
#         ProductCD
#         card1 - card6
#         addr1, addr2
#         P_emaildomain
#         R_emaildomain
#         M1 - M9
# 
# 
# **Identity Table**
# 
# The variables in this table encompass identity information, including network connection details (IP, ISP, Proxy, etc.) and digital signatures (UA/browser/os/version, etc.) associated with transactions.
# 
# These details are gathered by Vesta’s fraud protection system and digital security partners.
# (The field names are masked and a pairwise dictionary will not be provided for privacy protection and contractual agreement)
# 
#     **Categorical Features:**
# 
#     DeviceType
#     DeviceInfo
#     id_12 - id_38
# ## Exploratory Data Analysis on the data-sets - Part I
# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# **Exploratory Data Analysis (EDA) for the transaction file in the train data (transaction and identity)**
# ## Load the Data
# 
# 
# ---
# 
# 

# In[2]:


# Reading/Loading the data files (transaction and identity)
import pandas as pd
import numpy as np

train_transaction = pd.read_csv("train_transaction.csv")
train_identity = pd.read_csv("test_transaction.csv")


# ## EDA of **`train_transaction`** file
# 
# 
# ---
# 

# ### **BASIC EDA**
# 

# In[21]:


# printing the train_transaction df
train_transaction.head(5)


# In[22]:


# Information on the columns of train_transaction df
train_transaction.info()


# In[23]:


# data type of the specific columns
print(train_transaction.dtypes)


# In[24]:


# Basic statistical despcription of train_transaction df
train_transaction.describe()


# In[25]:


# Percentage of Null values of each column the data frame
null_columns = train_transaction.isna().sum()
#print(f"The dtype of null_columns is: '{null_columns.dtype()}'")
print(f"The columns with zero null values is: /n '{null_columns[null_columns==0]}'")
print(f"The number of columns with zero null values is: '{len(null_columns[null_columns==0])}' ")
print(f"The percentage of null values in the following column is : '{null_columns[null_columns>0]/train_transaction.shape[0]}'")
print(f"The number of columns with non-zero null values is :'{len(null_columns[null_columns>0])}' ")
print(max(null_columns[null_columns>0]/train_transaction.shape[0]))


# #### **Summary of Basic Eda**
# 
# 
# The `train_transaction` dataframe contains 590540 **(~590K)** entries/observations/rows and **394** columns with the **target** column **`isFraud`** having **no null** values. The target feature is represented as a numeric column with '0' representing a non-fraud transaction and '1' represeting a fraud transaction.
# 
# The target column also contains only 3.5% values that represent fradulent transaction of the total transactions. In other words we have only **~20,610 observations/rows** that portray the features that describe the **fradulent** transactions. The dataset that we have been presented with is highly imbalanced as expected due to the inherent nature of the low number of fraud transaction happening in real world compared to legally correct transation(s). **This implies we have to come up with ways to address the imbalance present in this dataset.**
# 
# This dataset has **376 columns** with **float64**, **4 columns** with **int64**,and **14 columns** with object data type. Among these columns, there are only 20 columns with no-null values and 374 columns with null values ranging from 2% to about 94%. Further this dataset also contains features obtained after performing **data/feature enrichment** (V1-V339) and hence it is difficult to find further details about many columns/features.
# 
# Hence, we proceed to conduct visual data analysis of this dataset to find any patterns/insights.
# 
# It is to be noted that the dataset contains many obfuscated features whose real world meaning is unobtainable and also the data is presented in groups of features (whose meaning is described in 'More Information about the Dataset heading) as described below:
# - TransactionID, isFraud, TransactionDT, TransactionAmt, ProductCD,
# - card1, card2, card3, card4, card5, card6,
# - addr1, addr2, dist1, dist2, P_emaildomain, R_emaildomain,
# - C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14,
# - D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15,
# - M1, M2, M3, M4, M5, M6, M7, M8, M9,
# - V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31, V32, V33, V34, V35, V36, V37, V38, V39, V40, V41, V42, V43, V44, V45, V46, V47, V48, V49, V50, V51, V52, V53, V54, V55, V56, V57, V58, V59, V60, V61, V62, V63, V64, V65, V66, V67, V68, V69, V70, V71, V72, V73, V74, V75, V76, V77, V78, V79, V80, V81, V82, V83, V84, V85, V86, V87, V88, V89, V90, V91, V92, V93, V94, V95 .... V337, V338, V339.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# ### VISUAL EDA

# In[26]:


# importing the libraries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

my_params = mpl.rcParams


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns

# Bar plot of the transaction amount to inspect the distribution of the feature
fig, ax = plt.subplots()
sns.histplot(data=train_transaction, x='TransactionAmt', hue='isFraud', bins=300, palette=["red", "black"], stat='percent')
ax.set_xlim(-10, 2000)
plt.title("Fig-1: Bar-plot of transaction amount differentiating fraud and non-fraud transactions")
plt.show()


# In[28]:


# Barplot of transactions through various card vendors (card4) differentiating fraudulent and true transactions
plt.figure(figsize=(10,5))
sns.histplot(data=train_transaction,x='card4',hue='isFraud',stat='percent')
plt.title("Fig-2: Bar-plot of transaction percentage through various card vendors with their Fraud transaction")
plt.xlabel('Card Type')
plt.show()


# In[29]:


# preprocess the datetime column
# get it down to day of week, hour
# hue by day of week
# countplot of 'isFraud'

train_transaction['Transaction_dow'] = np.floor((train_transaction['TransactionDT'] / (3600 * 24) - 1) % 7)
train_transaction['Transaction_hour'] = np.floor(train_transaction['TransactionDT'] / 3600) % 24

plt.figure(figsize=(10,5))
sns.histplot(data=train_transaction,x='Transaction_dow',hue='isFraud',stat='percent',palette=["red", "black"])
plt.title("Fig-3: Bar-plot of transaction percentage through various days of week with their Fraud transaction")
plt.xlabel('Day of week')
plt.show()


# In[32]:


plt.figure(figsize=(10,5))
sns.histplot(data=train_transaction,x='Transaction_hour',hue='isFraud',stat='percent',palette=["red", "black"])
plt.title("Fig-4: Bar-plot of transaction percentage through various hours with their Fraud transaction")
plt.xlabel('Hours')
plt.show()


# In[33]:


# count plots of the two classes
train_transaction['isFraud'].value_counts(normalize=True).plot(kind='bar')
plt.title("Fig-5: Relative counts of the real and fraud transactions in the data")


# In[39]:


# Box-plot of numerical features

num_column_lst = []
for icol in train_transaction.columns:
  if train_transaction[icol].dtypes in ['float64', 'int64']:
    num_column_lst.append(icol)
print(f"The columns that have numerical features are: '{num_column_lst}'")

"""
The column V1-V339 are data enriched dfeatures provided to us and we since we do not know what they represent
we won't be able to conduct feature engineering for outlier removal etc. Hence we need only the rest
42 columns which we can select by printing and selcting the respective columns in list.

TransactionID is just an identification and so a box plot of the same does not make sense. So we are not considering it
for the box plots.
"""

num_cols_lst = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1', 'dist2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15']
fig, axs = plt.subplots(nrows = 8, ncols=5, figsize = (150,100))
sns.set(font_scale = 4)
for name,ax in zip(num_cols_lst,axs.flatten()):
  sns.boxplot(data=train_transaction,x=name,ax=ax)
  # plt.show()
fig.tight_layout()
fig.suptitle('Figure-6: Box plot of non-enriched numerical variables',y=1.1,fontsize=160)
fig.subplots_adjust(top=1.06)


# In[42]:


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator

cat_column_lst = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(25, 25))
for name, ax in zip(cat_column_lst, axs.flatten()):
    sns.countplot(data=train_transaction, x=name, ax=ax, hue='isFraud')

    ticks = ax.get_xticks()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=15)
    ax.xaxis.set_major_locator(FixedLocator(ticks))

fig.tight_layout()
fig.suptitle('Figure-7: Bar-plot of Categorical variables', y=1.02, fontsize=26)

# Display the plot
plt.show()


# In[36]:


num_cols_lst = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1', 'dist2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15']


# In[43]:


# Bar-Plots of the numerical features(non-enriched) to examine the distribution of the feature

fig, axs = plt.subplots(nrows=10, ncols=4, figsize=(30, 40))
for name,ax in zip(num_cols_lst,axs.flatten()):
  sns.histplot(data=train_transaction,x=name,ax=ax,hue='isFraud',multiple='stack',stat='percent',bins=8)
  ax.tick_params(axis='x', labelsize=12)  # Adjust label size as needed
  ax.tick_params(axis='y', labelsize=12)

fig.tight_layout()
fig.suptitle('Figure-8: Distribution plot of numerical variables',y=1.02,fontsize=26)


# In[44]:


# Define the number of plots per figure and calculate the number of required figures
plots_per_figure = 10
num_figures = len(num_cols_lst) // plots_per_figure + (len(num_cols_lst) % plots_per_figure > 0)

for fig_index in range(num_figures):
    # Calculate the start and end indices for the columns to be plotted in the current figure
    start_idx = fig_index * plots_per_figure
    end_idx = start_idx + plots_per_figure
    # Get the subset of columns for the current figure
    cols_to_plot = num_cols_lst[start_idx:end_idx]

    # Create a new figure with the required number of subplots
    nrows = min(len(cols_to_plot), plots_per_figure)
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(15, 5 * nrows))  # Adjust figsize as needed

    # If only one row, axs is not an array, but we need it to be one for consistent indexing
    if nrows == 1:
        axs = [axs]

    for ax, name in zip(axs, cols_to_plot):
        sns.histplot(data=train_transaction, x=name, ax=ax, hue='isFraud', multiple='stack', stat='percent', bins=8)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_title(name)

    # Layout adjustments and titling the figure
    fig.tight_layout()
    sup_title = f'Figure-{fig_index + 8}: Distribution plot of numerical variables {start_idx + 1} to {end_idx}'
    fig.suptitle(sup_title, y=1.02, fontsize=26)

    plt.show()  # Show the current figure before moving on to the next


# #### **Summary of Visual EDA**
# 
# The transaction data primarily spans from 0 to 1200 currency units, with fraudulent activities mostly occurring within a range of a few to under 1000 currency units. Transactions from leading credit card companies like Visa, Mastercard, Discover, and American Express are included, with Visa transactions constituting around 60% of the entire training set, followed by Mastercard at approximately 30%. This distribution is also reflected in fraudulent activities, where Visa transactions are predominant, yet they have the lowest fraud-to-genuine transaction ratio among all providers. There's a noticeable pattern of fraud transactions peaking during early hours and late afternoon, with a dip in frequency around 6 to 12 hours. Conversely, no particular pattern emerged concerning the day of the week.
# 
# Upon examining the box plots for numerical features, notable outliers were detected in almost all variables except for a few, including card1, card2, card5, D2, D9, and D11. Therefore, we decided against outlier removal due to the data's volume, the underrepresentation of one class, and a large number of obscured features, which complicated the identification of an effective and efficient method for outlier treatment. It is important to highlight that transaction amounts exhibited significant outliers reaching up to approximately 5000 currency units, far exceeding the 90th percentile threshold of 1200 currency units.
# 
# Categorical variables were analyzed using bar plots colored by the target classes, and numerical variables were visualized using histograms to observe their distribution. None of the features were distinctly indicative of fraudulent transactions. The enigmatic nature of many variables further obscured the potential for clear conclusions from the box plots, bar plots, and histograms. Fraudulent transactions were dispersed across all unique categories and numerical bins, reflecting the problem's intricacy. The occurrence or absence of outliers in certain features did not correlate with the prevalence of fraud, as seen in variables like D9 and D11, where fraud was consistently distributed across bins, akin to features with significant outliers. No discernible distribution patterns were apparent that could offer insights for further analysis.
# 
# In summary, both basic and visual exploratory data analysis (EDA) of this transaction dataset unveil its complexity. Constructing a classification model to predict fraud is expected to be an intricate task that necessitates a methodical approach to EDA, coupled with advanced modeling techniques to accurately capture and predict fraudulent activities.

# ### BASIC EDA OF FRAUDULENT TRANSACTION DATA

# In[45]:


# Creation of DF with 'isfraud==true' data
isfraudt_df = train_transaction.loc[train_transaction['isFraud']==1]
isfraudt_df.head()


# In[46]:


# Information of the isfraud_df
isfraudt_df.info()


# In[47]:


# Basic statistical description of the isfraud_df
isfraudt_df.describe()


# In[48]:


## Percentage of Null values of each column the of the isfraud_df

isfraudt_null_columns = isfraudt_df.isna().sum()

print(f"The columns with zero null values is:  '{isfraudt_null_columns[isfraudt_null_columns==0]}'")
print(f"The number of columns with zero null values is: '{len(isfraudt_null_columns[isfraudt_null_columns==0])}' ")
print(f"The percentage of null values in the following column is : '{isfraudt_null_columns[isfraudt_null_columns>0]/train_transaction.shape[0]}'")
print(f"The number of columns with non-zero null values is :'{len(isfraudt_null_columns[isfraudt_null_columns>0])}' ")
print(max(isfraudt_null_columns[isfraudt_null_columns>0]/train_transaction.shape[0]))


# #### **Basic EDA Summary of Fraudulent Transactions**
# 
# The preliminary exploratory data analysis focused solely on fraudulent transactions reveals that while the count of features with or without missing values is similar to the overall dataset, the proportion of missing values is markedly lower—dropping from around 96% in the full dataset to roughly 3% in the fraud-only subset.
# 
# This notable decrease in missing values could be anticipated given the smaller volume of fraud transaction data. However, by assessing the percentage of missing values in both the comprehensive dataset and the subset encompassing only fraudulent transactions, it becomes evident that the subset offers a richer dataset for features associated with fraudulent activities.
# 

# ### VISUAL EDA OF FRADULENT TRANSACTIONS

# In[49]:


# Bar plot of the transaction amount to inspect the distribution of the feature
fig, ax = plt.subplots()
sns.histplot(data=isfraudt_df,x='TransactionAmt',bins=300)
ax.set_xlim(-10,2000)
plt.title("Fig-9: Bar-plot of the transaction amount of fraudulent transactions")
plt.show()


# In[50]:


# Barplot of fradulent transactions through various card vendors (card4)
plt.figure(figsize=(10,5))
sns.histplot(data=isfraudt_df,x='card4')
plt.title("Fig-10: Bar-plot of fraudulent transaction percentage of various card types")
plt.xlabel('Card Type')
plt.show()


# In[51]:


# Box-plot of 38 non-enriched numerical features of the fradulent transactions

num_cols_lst = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1', 'dist2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15']
print(num_cols_lst)

fig, axs = plt.subplots(nrows = 8, ncols=5, figsize = (150,100))
sns.set(font_scale = 4)
for name,ax in zip(num_cols_lst,axs.flatten()):
  sns.boxplot(data=isfraudt_df,x=name,ax=ax)
  # plt.show()
fig.tight_layout()
fig.suptitle('Fig-11: Box plot of selected numerical variables (non-enriched)of Fradulent transaction data',y=1.1,fontsize=160)
fig.subplots_adjust(top=1.06)


# In[52]:


# Bar-Plots of the 14 categorical features(non-enriched features) to examine the distribution of the various categories in a feature

print(cat_column_lst)

fig, axs = plt.subplots(nrows = 4, ncols=3, figsize = (25,25))
for name,ax in zip(cat_column_lst,axs.flatten()):
  sns.countplot(data=isfraudt_df, x=name, ax=ax)
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right',fontsize=15)

fig.tight_layout()
fig.suptitle('Fig-12: Bar-plot of Categorical variables of fraudenlent transaction data',y=1.02,fontsize=26)


# In[53]:


# Bar-Plots of the 42 numerical features(non-enriched) to examine the distribution of the feature

# print(num_column_lst)
print(num_cols_lst)

fig, axs = plt.subplots(nrows = 4, ncols=3, figsize = (25,25))
for name,ax in zip(num_cols_lst,axs.flatten()):
  sns.histplot(data=isfraudt_df,x=name,ax=ax,hue='isFraud',multiple='stack',stat='percent',bins=10)
  #ax.set_xticklabels(ax.get_xticklabels(minor=True,which='minor'),rotation=0,fontsize=15)

fig.tight_layout()
fig.suptitle('Figure-13: Distribution plot of numerical variables of the fradulent transaction data',y=1.02,fontsize=26)




# #### **Visual EDA Summary for Fraudulent Transactions**
# In the visual exploration of the fraudulent transactions, the box plots for numerical variables and bar plots for categorical variables offer a more detailed view, augmenting the hued visualizations seen in the analysis of the full dataset.
# 
# Focusing solely on the fraud subset highlights that a greater number of numerical features are free from or have fewer outliers compared to the full dataset. Yet, for interpretable features such as the transaction amount, the presence of outliers aligns with those observed in the overarching dataset. A closer inspection reveals a similarity in the outlier patterns across many features between the fraud-specific and complete datasets, which reinforces our decision to retain outliers, avoiding potential data loss and preserving the integrity of our fraud transaction insights.
# 
# Isolating one class for in-depth analysis proves beneficial in this context, especially as we delve into voluminous feature groups like the V-series. Analyzing these features within the context of fraudulent data may shed light on their utility, guiding decisions on feature retention, elimination, or further engineering.
# 
# The V-features, numbering 400, present a case in point. Breaking down this subset for detailed analysis could simplify the task, facilitating more informed feature engineering steps and contributing to a more nuanced understanding of the data's characteristics.

# ### EDA of V1 - V339

# In[54]:


# creating sub dataframe of the v-columns

train_transaction_v = train_transaction.iloc[:, 54:395]
train_transaction_v.head()


# In[55]:


# Bar chart of the V-columns based on the null values
null_columns_prcnt = train_transaction.isna().sum()/train_transaction.shape[0]
colsToPlot = ['V' + str(i) for i in range(1,340)]
plt.figure(figsize=(105,35))
null_columns_prcnt[colsToPlot].plot(kind='bar')
mpl.rcParams.update(my_params)
plt.title("Fig-14: Bar-plot of null values percentage on V-columns of transaction data",fontsize=45)
plt.xlabel('V columns')
plt.xticks(fontsize=18)
plt.yticks(fontsize=30)
plt.show()


# In[56]:


# Creating a dictionary of the groups of V-columns to create subplots of the groups to visualize null values
# Grouping the V-columns based on their percentage of null values

v_group_dict = {'grp1':['V' + str(i) for i in range(1,12)],
                'grp2':['V' + str(i) for i in range(12,35)],
                'grp3':['V' + str(i) for i in range(35,53)],
                'grp4':['V' + str(i) for i in range(53,75)],
                'grp5':['V' + str(i) for i in range(75,95)],
                'grp6':['V' + str(i) for i in range(95,138)],
                'grp7':['V' + str(i) for i in range(138,167)],
                'grp8':['V' + str(i) for i in range(167,217)],
                'grp9':['V' + str(i) for i in range(217,279)],
                'grp10':['V' + str(i) for i in range(279,322)],
                'grp11':['V' + str(i) for i in range(322,340)]}

fig, axs = plt.subplots(nrows = 3, ncols=4, figsize = (105,40))
for name,ax in zip(v_group_dict.keys(),axs.flatten()):
  null_columns_prcnt[v_group_dict[name]].plot(ax=ax,kind='bar')
  ax.tick_params(axis='both', which='major', labelsize=37)

fig.tight_layout()
fig.suptitle('Fig-15: Bar plot of the V-columns columns separated by groups based on their null value strcture',y=1.1,fontsize=60)
fig.subplots_adjust(top=1.06)


# In[57]:


# correlation each v-columns within the group


# In[83]:


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(corrMatrix, corrThresh):
    au_corr = corrMatrix.abs().unstack()
    labels_to_drop = get_redundant_pairs(corrMatrix)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[au_corr>corrThresh]

def plotCorrMatrixAndGetColsToDrop(df,figsize,threshold,columnsToCorr):
  plt.figure(figsize=figsize)
  grp_correlationMatrix = df[columnsToCorr].corr()
  sns.heatmap(grp_correlationMatrix, cmap='RdBu_r', annot=True, center=0.0)

  #compute the columns to drop from a given list of columns
  high_corr_pairs = get_top_abs_correlations(grp_correlationMatrix, threshold)

  upper = grp_correlationMatrix.where(np.triu(np.ones(grp_correlationMatrix.shape), k=1).astype(bool))
  to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

  return high_corr_pairs,grp_correlationMatrix,to_drop

  # Correlation of grp-1 with the target variable and within V columns
corrThresh = 0.65
grp1_high_corr_pairs,grp1_correlationMatrix,grp1_todrop = plotCorrMatrixAndGetColsToDrop(train_transaction,(15,15),corrThresh,v_group_dict['grp1']+['isFraud'])


# In[68]:


def plotCorrMatrix(df, columnsToCorr, figsize=(10, 8)):
    '''Plot the correlation matrix for a given list of columns.'''
    corrMatrix = df[columnsToCorr].corr()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corrMatrix, dtype=np.bool_))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corrMatrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, annot=True, cbar_kws={"shrink": .5},
                linewidths=.5, fmt=".2f", annot_kws={"size": 9})

    # Improve the visibility of the heatmap
    for text in ax.texts:
        text_content = text.get_text()
        if text_content:  # This checks if the text is not empty
            if abs(float(text_content)) < 0.1:  # Threshold for visibility
                text.set_text('')
            else:
                text.set_text(text_content)
                text.set_color('black' if abs(float(text_content)) > 0.5 else 'white')

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

# Assuming train_transaction and v_group_dict are defined and contain the necessary data
# Here we plot only a subset of variables for better visibility
selected_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'isFraud']
plotCorrMatrix(train_transaction, selected_columns, figsize=(12, 10))


# In[66]:


# extract the results from grp1
retain_cols = list(set(v_group_dict['grp1']) - set(grp1_todrop))
v_grp1_dict_results = {'Columns':v_group_dict['grp1'],
               'num columns':len(v_group_dict['grp1']),
               'Dropping Columns':grp1_todrop,
               'Num Columns Dropped':len(grp1_todrop),
               'Retain Columns':retain_cols,
               'Num Columns Retained': len(retain_cols)}
v_grp1_dict_results


# 1. Drop the required columns
# 2. Subset of the V group 1 columns to keep
# 3. Columns to drop from group 1 - V11,V5,V8,V6,V3

# In[69]:


grp1_todrop


# In[84]:


# correlation of group 2 and group 4 v-columns within the group
# plt.figure(figsize=(35,35))
# grp2_correlationMatrix=train_transaction[v_group_dict['grp2']+v_group_dict['grp4']].corr()
grp2n4_high_corr_pairs, grp2n4_correlationMatrix, grp2n4_todrop = plotCorrMatrixAndGetColsToDrop(train_transaction,(35,35),corrThresh,v_group_dict['grp2']+v_group_dict['grp4']+['isFraud'])


# In[85]:


pd.options.display.max_rows = 4000


# In[86]:


# extract the results from grp2 and grp4

vgrp2n4_retain_cols = list(set(v_group_dict['grp2']+v_group_dict['grp4']) - set(grp2n4_todrop))
v_grp2n4_dict_results = {'Columns':(v_group_dict['grp2']+v_group_dict['grp4']),
               'num columns':len(v_group_dict['grp2']+v_group_dict['grp4']),
               'Dropping Columns':grp2n4_todrop,
               'Num Columns Dropped':len(grp2n4_todrop),
               'Retain Columns':vgrp2n4_retain_cols,
               'Num Columns Retained': len(vgrp2n4_retain_cols)}
v_grp2n4_dict_results


# In[87]:


# correlation of group 3 v-columns within the group
# plt.figure(figsize=(15,15))
# sns.heatmap(train_transaction[v_group_dict['grp3']].corr(), cmap='RdBu_r', annot=True, center=0.0)
grp3_high_corr_pairs, grp3_correlationMatrix, grp3_todrop = plotCorrMatrixAndGetColsToDrop(train_transaction,(15,15),corrThresh,v_group_dict['grp3']+['isFraud'])


# In[88]:


# extract the results from grp3

vgrp3_retain_cols = list(set(v_group_dict['grp3']) - set(grp3_todrop))
v_grp3_dict_results = {'Columns':(v_group_dict['grp3']),
               'num columns':len(v_group_dict['grp3']),
               'Dropping Columns':grp3_todrop,
               'Num Columns Dropped':len(grp3_todrop),
               'Retain Columns':vgrp3_retain_cols,
               'Num Columns Retained': len(vgrp3_retain_cols)}
v_grp3_dict_results


# In[89]:


# correlation of group 5 v-columns within the group
# plt.figure(figsize=(15,15))
# sns.heatmap(train_transaction[v_group_dict['grp5']].corr(), cmap='RdBu_r', annot=True, center=0.0)
grp5_high_corr_pairs, grp5_correlationMatrix, grp5_todrop = plotCorrMatrixAndGetColsToDrop(train_transaction,(15,15),corrThresh,v_group_dict['grp5']+['isFraud'])


# In[90]:


# extract the results from grp5

vgrp5_retain_cols = list(set(v_group_dict['grp5']) - set(grp5_todrop))
v_grp5_dict_results = {'Columns':(v_group_dict['grp5']),
               'num columns':len(v_group_dict['grp5']),
               'Dropping Columns':grp5_todrop,
               'Num Columns Dropped':len(grp5_todrop),
               'Retain Columns':vgrp5_retain_cols,
               'Num Columns Retained': len(vgrp5_retain_cols)}
v_grp5_dict_results


# In[91]:


# correlation of group 6 v-columns within the group
# plt.figure(figsize=(35,35))
# sns.heatmap(train_transaction[v_group_dict['grp6']].corr(), cmap='RdBu_r', annot=True, center=0.0)
grp6_high_corr_pairs, grp6_correlationMatrix, grp6_todrop = plotCorrMatrixAndGetColsToDrop(train_transaction,(35,35),corrThresh,v_group_dict['grp6']+['isFraud'])


# In[92]:


# extract the results from grp6

vgrp6_retain_cols = list(set(v_group_dict['grp6']) - set(grp6_todrop))
v_grp6_dict_results = {'Columns':(v_group_dict['grp6']),
               'num columns':len(v_group_dict['grp6']),
               'Dropping Columns':grp6_todrop,
               'Num Columns Dropped':len(grp6_todrop),
               'Retain Columns':vgrp6_retain_cols,
               'Num Columns Retained': len(vgrp6_retain_cols)}
v_grp6_dict_results


# In[93]:


# correlation of group 7 and 11 v-columns within the group
# plt.figure(figsize=(35,35))
# sns.heatmap(train_transaction[v_group_dict['grp7']+v_group_dict['grp11']].corr(), cmap='RdBu_r', annot=True, center=0.0)
grp7n11_high_corr_pairs, grp7n11_correlationMatrix, grp7n11_todrop = plotCorrMatrixAndGetColsToDrop(train_transaction,(35,35),corrThresh,v_group_dict['grp7']+v_group_dict['grp11']+['isFraud'])


# In[94]:


# extract the results from grp7 and grp11

vgrp7n11_retain_cols = list(set(v_group_dict['grp7']+v_group_dict['grp11']) - set(grp7n11_todrop))
v_grp7n11_dict_results = {'Columns':(v_group_dict['grp7']+v_group_dict['grp11']),
               'num columns':len(v_group_dict['grp7']+v_group_dict['grp11']),
               'Dropping Columns':grp7n11_todrop,
               'Num Columns Dropped':len(grp7n11_todrop),
               'Retain Columns':vgrp7n11_retain_cols,
               'Num Columns Retained': len(vgrp7n11_retain_cols)}
v_grp7n11_dict_results


# In[95]:


# correlation of group 8 v-columns within the group
# plt.figure(figsize=(35,35))

grp8_high_corr_pairs, grp8_correlationMatrix, grp8_todrop = plotCorrMatrixAndGetColsToDrop(train_transaction,(35,35),corrThresh,v_group_dict['grp8']+['isFraud'])


# In[96]:


# extract the results from grp8

vgrp8_retain_cols = list(set(v_group_dict['grp8']) - set(grp8_todrop))
v_grp8_dict_results = {'Columns':(v_group_dict['grp8']),
               'num columns':len(v_group_dict['grp8']),
               'Dropping Columns':grp8_todrop,
               'Num Columns Dropped':len(grp8_todrop),
               'Retain Columns':vgrp8_retain_cols,
               'Num Columns Retained': len(vgrp8_retain_cols)}
v_grp8_dict_results


# In[97]:


# correlation of group 9 v-columns within the group
# plt.figure(figsize=(35,35))

grp9_high_corr_pairs, grp9_correlationMatrix, grp9_todrop = plotCorrMatrixAndGetColsToDrop(train_transaction,(35,35), corrThresh, v_group_dict['grp9']+['isFraud'])


# In[98]:


# extract the results from grp9

vgrp9_retain_cols = list(set(v_group_dict['grp9']) - set(grp9_todrop))
v_grp9_dict_results = {'Columns':(v_group_dict['grp9']),
               'num columns':len(v_group_dict['grp9']),
               'Dropping Columns':grp9_todrop,
               'Num Columns Dropped':len(grp9_todrop),
               'Retain Columns':vgrp9_retain_cols,
               'Num Columns Retained': len(vgrp9_retain_cols)}
v_grp9_dict_results


# In[99]:


# correlation of group 10 v-columns within the group
# plt.figure(figsize=(35,35))

grp10_high_corr_pairs, grp10_correlationMatrix, grp10_todrop = plotCorrMatrixAndGetColsToDrop(train_transaction,(35,35), corrThresh, v_group_dict['grp10']+['isFraud'])


# In[100]:


# extract the results from grp10

vgrp10_retain_cols = list(set(v_group_dict['grp10']) - set(grp10_todrop))
v_grp10_dict_results = {'Columns':(v_group_dict['grp10']),
               'num columns':len(v_group_dict['grp10']),
               'Dropping Columns':grp10_todrop,
               'Num Columns Dropped':len(grp10_todrop),
               'Retain Columns':vgrp10_retain_cols,
               'Num Columns Retained': len(vgrp10_retain_cols)}
v_grp10_dict_results


# #### **Summary of EDA of V1- V339**
# 
# Based on the analysis conducted on the complete dataset and fraudulent dataset alone, we subsetted the data consisting of features enriched based on their groups and started with the V-group first.
# 
# As these columns contained many null values, we analyzed the 11 subsetted groups of features for their null value counts. Visual analysis portrayed these features exhibiting null value counts as groups. Thus, we split the V-columns into groups based on their null values through visual inspection. These subgroups of V-columns were further analyzed for any additional details regarding their null structure. Since there wasn't more information that could be derived from these subgroups, we proceeded to the next step of the analysis.
# 
# In the next step, we analyzed each individual subgroup of the V-column, constructed based on their null structure in the previous step, for possible correlations among the features within that subgroup.
# 
# We observed that many features were correlated within each subgroup among all the 11 subgroups. Hence, we set a threshold value of 0.65 as the correlation coefficient to drop columns exhibiting correlation beyond this threshold.
# 
# The number of V-columns retained at every step of the correlation analysis would be concatenated and further analyzed in the following steps.
# 
# Additionally, following the same approach, we attempted to find if there were V-columns portraying similar null structure, as the null value percentage was very low in the subdataset containing only fraudulent data. This could be beneficial in finding any patterns of similarity or difference in null structure that could be more prominent or unique to the fraudulent data alone.

# ### EDA of V1-V339 of Fraudulent dataframe

# In[101]:


# Bar chart of the V-columns based on the null values

isfraudt_null_columns_prcnt = isfraudt_df.isna().sum()/isfraudt_df.shape[0]
isfraudt_colsToPlot = ['V' + str(i) for i in range(1,340)]
plt.figure(figsize=(105,35))
mpl.rcParams.update(my_params)
isfraudt_null_columns_prcnt[isfraudt_colsToPlot].plot(kind='bar')
plt.title("Fig-16: Bar-plot of null columns percentage of V-columns of Fraud transaction data",fontsize=48)
plt.xlabel('Null-value percentage',fontsize = 40)
plt.xticks(fontsize=18)
plt.yticks(fontsize=30)
plt.show()


# In[102]:


# Creating a dictionary of the groups of V-columns to create subplots of the groups to visualize null values
# Grouping the V-columns based on their percentage of null values

isfraudt_v_group_dict = {'ftgrp1':['V' + str(i) for i in range(1,12)],
                'ftgrp2':['V' + str(i) for i in range(12,35)],
                'ftgrp3':['V' + str(i) for i in range(35,53)],
                'ftgrp4':['V' + str(i) for i in range(53,75)],
                'ftgrp5':['V' + str(i) for i in range(75,95)],
                'ftgrp6':['V' + str(i) for i in range(95,138)],
                'ftgrp7':['V' + str(i) for i in range(138,167)],
                'ftgrp8':['V' + str(i) for i in range(167,217)],
                'ftgrp9':['V' + str(i) for i in range(217,279)],
                'ftgrp10':['V' + str(i) for i in range(279,322)],
                'ftgrp11':['V' + str(i) for i in range(322,340)]}

fig, axs = plt.subplots(nrows = 3, ncols=4, figsize = (105,40))
for name,ax in zip(isfraudt_v_group_dict.keys(),axs.flatten()):
  isfraudt_null_columns_prcnt[isfraudt_v_group_dict[name]].plot(ax=ax,kind='bar')
  ax.tick_params(axis='both', which='major', labelsize=37)

fig.tight_layout()
fig.suptitle('Fig-17: Bar plot of the V-columns of fraudulent data separated by groups based ont their null structure',y=1.1,fontsize=70)
fig.subplots_adjust(top=1.06)


# In[102]:





# v138 - v166, v322-v339
# 

# In[103]:


# Correlation of grp-1 with the target variable and within V columns
corrThresh = 0.65
ftgrp1_high_corr_pairs, ftgrp1_correlationMatrix, ftgrp1_todrop = plotCorrMatrixAndGetColsToDrop(isfraudt_df,(15,15),corrThresh,isfraudt_v_group_dict['ftgrp1'])


# The column V1 for fraud transaction contains only two values (NAN and 1.0) and hence the correlation of that column with any other column is resulting in NAN. Also we choose to retain this column irerspective of the previous cell's analysis choice we made as we notice the V1 column is significanlty different for the fraud transaction than the non-fraud transaction data indicating that it could be representing some information that distinguishes the two classes.  

# In[104]:


# extract the results from ftgrp1

ftgrp1_retain_cols = list(set(isfraudt_v_group_dict['ftgrp1']) - set(ftgrp1_todrop))
ftv_grp1_dict_results = {'Columns':(isfraudt_v_group_dict['ftgrp1']),
               'num columns':len(isfraudt_v_group_dict['ftgrp1']),
               'Dropping Columns':ftgrp1_todrop,
               'Num Columns Dropped':len(ftgrp1_todrop),
               'Retain Columns':ftgrp1_retain_cols,
               'Num Columns Retained': len(ftgrp1_retain_cols)}
ftv_grp1_dict_results


# In[105]:


# correlation of group 2 v-columns within the group
# plt.figure(figsize=(35,35))
# grp2_correlationMatrix=train_transaction[v_group_dict['grp2']].corr()

ftgrp2_high_corr_pairs, ftgrp2_correlationMatrix, ftgrp2_todrop = plotCorrMatrixAndGetColsToDrop(isfraudt_df,(35,35),corrThresh,isfraudt_v_group_dict['ftgrp2']+['isFraud'])


# In[106]:


# extract the results from ftgrp2

ftgrp2_retain_cols = list(set(isfraudt_v_group_dict['ftgrp2']) - set(ftgrp2_todrop))
ftv_grp2_dict_results = {'Columns':(isfraudt_v_group_dict['ftgrp2']),
               'num columns':len(isfraudt_v_group_dict['ftgrp2']),
               'Dropping Columns':ftgrp2_todrop,
               'Num Columns Dropped':len(ftgrp2_todrop),
               'Retain Columns':ftgrp2_retain_cols,
               'Num Columns Retained': len(ftgrp2_retain_cols)}
ftv_grp2_dict_results


# In[107]:


# correlation of group 3 v-columns within the group
ftgrp3_high_corr_pairs, ftgrp3_correlationMatrix, ftgrp3_todrop = plotCorrMatrixAndGetColsToDrop(isfraudt_df,(15,15),corrThresh,isfraudt_v_group_dict['ftgrp3'])


# In[108]:


# extract the results from ftgrp3

ftgrp3_retain_cols = list(set(isfraudt_v_group_dict['ftgrp3']) - set(ftgrp3_todrop))
ftv_grp3_dict_results = {'Columns':(isfraudt_v_group_dict['ftgrp3']),
               'num columns':len(isfraudt_v_group_dict['ftgrp3']),
               'Dropping Columns':ftgrp3_todrop,
               'Num Columns Dropped':len(ftgrp3_todrop),
               'Retain Columns':ftgrp3_retain_cols,
               'Num Columns Retained': len(ftgrp3_retain_cols)}
ftv_grp3_dict_results


# In[109]:


# correlation of group 4 v-columns within the group
ftgrp4_high_corr_pairs, ftgrp4_correlationMatrix, ftgrp4_todrop = plotCorrMatrixAndGetColsToDrop(isfraudt_df,(15,15),corrThresh,isfraudt_v_group_dict['ftgrp4'])


# In[110]:


# extract the results from ftgrp4

ftgrp4_retain_cols = list(set(isfraudt_v_group_dict['ftgrp4']) - set(ftgrp4_todrop))
ftv_grp4_dict_results = {'Columns':(isfraudt_v_group_dict['ftgrp4']),
               'num columns':len(isfraudt_v_group_dict['ftgrp4']),
               'Dropping Columns':ftgrp4_todrop,
               'Num Columns Dropped':len(ftgrp4_todrop),
               'Retain Columns':ftgrp4_retain_cols,
               'Num Columns Retained': len(ftgrp4_retain_cols)}
ftv_grp4_dict_results


# In[111]:


# correlation of group 5 v-columns within the group
ftgrp5_high_corr_pairs, ftgrp5_correlationMatrix, ftgrp5_todrop = plotCorrMatrixAndGetColsToDrop(isfraudt_df,(15,15),corrThresh,isfraudt_v_group_dict['ftgrp5'])


# In[112]:


# extract the results from ftgrp5

ftgrp5_retain_cols = list(set(isfraudt_v_group_dict['ftgrp5']) - set(ftgrp5_todrop))
ftv_grp5_dict_results = {'Columns':(isfraudt_v_group_dict['ftgrp5']),
               'num columns':len(isfraudt_v_group_dict['ftgrp5']),
               'Dropping Columns':ftgrp5_todrop,
               'Num Columns Dropped':len(ftgrp5_todrop),
               'Retain Columns':ftgrp5_retain_cols,
               'Num Columns Retained': len(ftgrp5_retain_cols)}
ftv_grp5_dict_results


# In[113]:


# correlation of group 6 v-columns within the group
ftgrp6_high_corr_pairs, ftgrp6_correlationMatrix, ftgrp6_todrop = plotCorrMatrixAndGetColsToDrop(isfraudt_df,(35,35),corrThresh,isfraudt_v_group_dict['ftgrp6'])


# In[114]:


# extract the results from ftgrp6

ftgrp6_retain_cols = list(set(isfraudt_v_group_dict['ftgrp6']) - set(ftgrp6_todrop))
ftv_grp6_dict_results = {'Columns':(isfraudt_v_group_dict['ftgrp6']),
               'num columns':len(isfraudt_v_group_dict['ftgrp6']),
               'Dropping Columns':ftgrp6_todrop,
               'Num Columns Dropped':len(ftgrp6_todrop),
               'Retain Columns':ftgrp6_retain_cols,
               'Num Columns Retained': len(ftgrp6_retain_cols)}
ftv_grp6_dict_results


# In[115]:


# correlation of group 7 v-columns within the group
ftgrp7_high_corr_pairs, ftgrp7_correlationMatrix, ftgrp7_todrop = plotCorrMatrixAndGetColsToDrop(isfraudt_df,(35,35),corrThresh,isfraudt_v_group_dict['ftgrp7'])


# In[116]:


# extract the results from ftgrp7

ftgrp7_retain_cols = list(set(isfraudt_v_group_dict['ftgrp7']) - set(ftgrp7_todrop))
ftv_grp7_dict_results = {'Columns':(isfraudt_v_group_dict['ftgrp7']),
               'num columns':len(isfraudt_v_group_dict['ftgrp7']),
               'Dropping Columns':ftgrp7_todrop,
               'Num Columns Dropped':len(ftgrp7_todrop),
               'Retain Columns':ftgrp7_retain_cols,
               'Num Columns Retained': len(ftgrp7_retain_cols)}
ftv_grp7_dict_results


# In[117]:


# correlation of group 8 v-columns within the group
ftgrp8_high_corr_pairs, ftgrp8_correlationMatrix, ftgrp8_todrop = plotCorrMatrixAndGetColsToDrop(isfraudt_df,(35,35),corrThresh,isfraudt_v_group_dict['ftgrp8'])


# In[118]:


# extract the results from ftgrp8

ftgrp8_retain_cols = list(set(isfraudt_v_group_dict['ftgrp8']) - set(ftgrp8_todrop))
ftv_grp8_dict_results = {'Columns':(isfraudt_v_group_dict['ftgrp8']),
               'num columns':len(isfraudt_v_group_dict['ftgrp8']),
               'Dropping Columns':ftgrp8_todrop,
               'Num Columns Dropped':len(ftgrp8_todrop),
               'Retain Columns':ftgrp8_retain_cols,
               'Num Columns Retained': len(ftgrp8_retain_cols)}
ftv_grp8_dict_results


# In[119]:


# correlation of group 9 v-columns within the group
ftgrp9_high_corr_pairs, ftgrp9_correlationMatrix, ftgrp9_todrop = plotCorrMatrixAndGetColsToDrop(isfraudt_df,(35,35),corrThresh,isfraudt_v_group_dict['ftgrp9'])


# In[120]:


# extract the results from ftgrp9

ftgrp9_retain_cols = list(set(isfraudt_v_group_dict['ftgrp9']) - set(ftgrp9_todrop))
ftv_grp9_dict_results = {'Columns':(isfraudt_v_group_dict['ftgrp9']),
               'num columns':len(isfraudt_v_group_dict['ftgrp9']),
               'Dropping Columns':ftgrp9_todrop,
               'Num Columns Dropped':len(ftgrp9_todrop),
               'Retain Columns':ftgrp9_retain_cols,
               'Num Columns Retained': len(ftgrp9_retain_cols)}
ftv_grp9_dict_results


# In[121]:


# correlation of group 10 v-columns within the group
ftgrp10_high_corr_pairs, ftgrp10_correlationMatrix, ftgrp10_todrop = plotCorrMatrixAndGetColsToDrop(isfraudt_df,(35,35),corrThresh,isfraudt_v_group_dict['ftgrp10'])


# In[122]:


# extract the results from ftgrp10

ftgrp10_retain_cols = list(set(isfraudt_v_group_dict['ftgrp10']) - set(ftgrp10_todrop))
ftv_grp10_dict_results = {'Columns':(isfraudt_v_group_dict['ftgrp10']),
               'num columns':len(isfraudt_v_group_dict['ftgrp10']),
               'Dropping Columns':ftgrp10_todrop,
               'Num Columns Dropped':len(ftgrp10_todrop),
               'Retain Columns':ftgrp10_retain_cols,
               'Num Columns Retained': len(ftgrp10_retain_cols)}
ftv_grp10_dict_results


# In[123]:


# correlation of group 11 v-columns within the group
ftgrp11_high_corr_pairs, ftgrp11_correlationMatrix, ftgrp11_todrop = plotCorrMatrixAndGetColsToDrop(isfraudt_df,(15,15),corrThresh,isfraudt_v_group_dict['ftgrp11'])


# In[124]:


# extract the results from ftgrp11

ftgrp11_retain_cols = list(set(isfraudt_v_group_dict['ftgrp11']) - set(ftgrp11_todrop))
ftv_grp11_dict_results = {'Columns':(isfraudt_v_group_dict['ftgrp11']),
               'num columns':len(isfraudt_v_group_dict['ftgrp11']),
               'Dropping Columns':ftgrp11_todrop,
               'Num Columns Dropped':len(ftgrp11_todrop),
               'Retain Columns':ftgrp11_retain_cols,
               'Num Columns Retained': len(ftgrp11_retain_cols)}
ftv_grp11_dict_results


# #### **Summary of EDA of  Fraudulent V1-V339**
# 
# We subsetted the V-group columns of the fraudulent dataset and analyzed the null structure of these columns. V-columns with similar null structures were grouped together and further analyzed to find any more granularity that could be utilized. However, we couldn't find any further details, so we proceeded to conduct correlation analysis of subgroups of V-columns with similar null structures. We set the correlation threshold to 0.65 and retained only the non-correlated features of each subgroup whose correlation was less than 0.65.
# 
# As already discovered in the EDA of the whole dataset and the fraudulent dataset, the major difference in null structure between the two subdatasets was the percentage of null values, which was also true for the V-columns. Upon closer examination, we found that although the basic pattern of V-group columns being present as clusters exhibiting similar null values persisted, the percentage of null values exhibited by V-columns among the two datasets varies.
# 
# This difference could have implications in the columns retained by this dataset, which could vary from the columns retained in the complete dataset after the correlation analysis.

# ### Reduced V - Columns

# In[125]:


# compare the retained list of train_transaction and isfraudt_df
# find the columns len of retained columns in each group
# find the columns from train_transaction that is not presentin isfraudt_df
# find columns from isfraudt_df that is not present in train_transaction

# Retain all columns from both


# In[126]:


# Total reduced columns from the train_transaction dataframe

vt_reduced_col = v_grp1_dict_results['Retain Columns']+ v_grp2n4_dict_results['Retain Columns'] + v_grp3_dict_results['Retain Columns']+\
                v_grp5_dict_results['Retain Columns'] + v_grp6_dict_results['Retain Columns'] + v_grp7n11_dict_results['Retain Columns']+\
                v_grp8_dict_results['Retain Columns'] + v_grp9_dict_results['Retain Columns'] + v_grp10_dict_results['Retain Columns']
print(len(vt_reduced_col))
vt_reduced_col


# In[127]:


# Total reduced columns from the sub-dataframe containng only fraud data

vfraud_reduced_col = ftv_grp1_dict_results['Retain Columns'] + ftv_grp2_dict_results['Retain Columns'] + ftv_grp3_dict_results['Retain Columns'] +\
                     ftv_grp4_dict_results['Retain Columns'] + ftv_grp5_dict_results['Retain Columns'] + ftv_grp6_dict_results['Retain Columns'] +\
                     ftv_grp7_dict_results['Retain Columns'] + ftv_grp8_dict_results['Retain Columns'] + ftv_grp9_dict_results['Retain Columns'] +\
                     ftv_grp10_dict_results['Retain Columns'] + ftv_grp11_dict_results['Retain Columns']
print(len(vfraud_reduced_col))
vfraud_reduced_col


# In[128]:


# Columns present in one group and not the other

# columns present in the reduced v-columns from the train_transaction df but not in fraud data
print(list(set(vt_reduced_col)-set(vfraud_reduced_col)))
print(len(list(set(vt_reduced_col)-set(vfraud_reduced_col))))

# columns present in the reduced v-columns from the fraud df but not in train_transaction df
print(list(set(vfraud_reduced_col)-set(vt_reduced_col)))
print(len(list(set(vfraud_reduced_col)-set(vt_reduced_col))))


# In[129]:


# Total reduced columns to take and sorting it

temp = np.unique(vt_reduced_col + vfraud_reduced_col)
V_reduced = []
colNumbers = [int(i_col.split('V')[1]) for i_col in temp]
colNumbers.sort()
V_reduced = ['V' + str(i_col) for i_col in colNumbers]
print(len(V_reduced))
V_reduced


#  **Summary of Reduced V-columns**
# 
# The V-columns retained from each subgroup of each dataset were concatenated to yield the reduced V-list for that specific dataset. Hence, we concatenated two sets of subgroups of V-columns from the previous step, giving us two V-reduced groups (consisting of a reduced number of V-features selected based on correlation; 94 features for the whole dataset and 90 for the fraud dataset).
# 
# We then identified the features from the reduced dataset that were unique to each group. Approximately 29 features were unique to the reduced group from the whole dataset, and 25 were unique to the reduced list from the fraud dataset.
# 
# Subsequently, we gathered all the columns that were unique to the V-reduced group of both datasets and all the columns that were present in the reduced group of both datasets. This resulted in approximately ~120 V-group features in the V-reduced features list.
# 

# #### Analysis of reduced V-columns

# In[130]:


# Make SNS heat map and take out columns that are .9 and above.
# Retain the column from the pair that is also present in the isfraudt_df list
corrThresh=.8
V_reduced_high_corr_pairs, V_reduced_correlationMatrix,V_reduced_todrop = plotCorrMatrixAndGetColsToDrop(train_transaction,(35,35),corrThresh,V_reduced)


# In[131]:


# extract the results from V_reduced columns

V_reduced_retain_cols = list(set(train_transaction[V_reduced]) - set(V_reduced_todrop))
V_reduced_dict_results = {'Columns':(train_transaction[V_reduced]),
               'num columns':len(train_transaction[V_reduced]),
               'Dropping Columns':V_reduced_todrop,
               'Num Columns Dropped':len(V_reduced_todrop),
               'Retain Columns':V_reduced_retain_cols,
               'Num Columns Retained': len(V_reduced_retain_cols)}
V_reduced_dict_results


# ### EDA of C-Columns

# In[152]:


# Creating a dictionary of the groups of c-columns to create subplots of the groups to visualize null values
# Grouping the c-columns based on their percentage of null values

corrThresh = 0.65
c_group_dict = {'ctgrp':['C' + str(i) for i in range(1,15)]}


# In[153]:


# Correlation of C-grp columns with the target variable and within V columns
ctgrp_high_corr_pairs, ctgrp_correlationMatrix, ctgrp_todrop = plotCorrMatrixAndGetColsToDrop(train_transaction,(15,15),corrThresh,c_group_dict['ctgrp']+['isFraud'])


# In[154]:


# extract the results from C grp columns

c_retain_cols = list(set(c_group_dict['ctgrp']) - set(ctgrp_todrop))
c_group_dict_results = {'Columns':c_group_dict['ctgrp'],
               'num columns':len(c_group_dict['ctgrp']),
               'Dropping Columns':ctgrp_todrop,
               'Num Columns Dropped':len(ctgrp_todrop),
               'Retain Columns':c_retain_cols,
               'Num Columns Retained': len(c_retain_cols)}
c_group_dict_results


# In[155]:


# look at c-9  not selected eventhough it is not much corelated # no null in c.


# #### Summary of EDA of C-columns

# We performed corelation analysis and set the threshold to 0.65 and retained columns that were not corerelated and whose corerelation coefficients were less than 0.65.
# 
# We found that we could reatin only 3 column out of the 14 features that contained 11 hughly corerelatedd features.

# ### EDA of D-Columns

# In[156]:


# Bar chart of the D-columns based on the null values

dCol_null_columns_prcnt = train_transaction.isna().sum()/train_transaction.shape[0]
dCol_colsToPlot = ['D' + str(i) for i in range(1,16)]
plt.figure(figsize=(105,35))
dCol_null_columns_prcnt[dCol_colsToPlot].plot(kind='bar')
plt.title("Fig-18: Bar-plot of percentage null values of D-group columns in train-transaction data",fontsize=48)
plt.xlabel('Null-value percentage',fontsize=48)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()


# In[157]:


# Creating a dictionary of the groups of V-columns to create subplots of the groups to visualize null values
# Grouping the V-columns based on their percentage of null values

corrThresh = 0.65
d_group_dict = {'dtgrp':['D' + str(i) for i in range(1,16)]}


# In[158]:


# Correlation of grp-1 with the target variable and within V columns
corrThresh = 0.65
dtgrp1_high_corr_pairs, dtgrp1_correlationMatrix, dtgrp1_todrop = plotCorrMatrixAndGetColsToDrop(train_transaction,(15,15),corrThresh,d_group_dict['dtgrp'])


# In[159]:


# extract the results from D grp columns

d_retain_cols = list(set(d_group_dict['dtgrp']) - set(dtgrp1_todrop))
d_group_dict_results = {'Columns':d_group_dict['dtgrp'],
               'num columns':len(d_group_dict['dtgrp']),
               'Dropping Columns':dtgrp1_todrop,
               'Num Columns Dropped':len(dtgrp1_todrop),
               'Retain Columns':d_retain_cols,
               'Num Columns Retained': len(d_retain_cols)}
d_group_dict_results


# #### Summary of EDA of D-columns

# After performing correlation analysis with the D-group columns we found that out of the 15 features present, 7 features were highly corerelated and hence we could retain only the remaining 8 features whose coefficients of corerelation was less than 0.65.

# ### EDA of M-Columns

# In[144]:


# Bar chart of the M-columns based on the null values

mCol_null_columns_prcnt = train_transaction.isna().sum()/train_transaction.shape[0]
mCol_colsToPlot = ['M' + str(i) for i in range(1,10)]
plt.figure(figsize=(105,35))
mCol_null_columns_prcnt[mCol_colsToPlot].plot(kind='bar')
plt.title("Fig-19: Bar-plot of percentage null values of M-group columns in train-transaction data",fontsize=48)
plt.xlabel('Null-value percentage',fontsize=48)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()


# In[145]:


# Creating a dictionary of the groups of M-columns to create subplots of the groups to visualize null values
# Grouping the M-columns based on their percentage of null values

corrThresh = 0.60
m_group_dict = {'mtgrp':['M' + str(i) for i in range(1,10)]}


# In[146]:


# Label encoding the M-group columns

mp = {'F':0,'T':1,'M0':0,'M1':1,'M2':2}
for c in m_group_dict['mtgrp']:
  train_transaction[c] = train_transaction[c].map(mp)
train_transaction[m_group_dict['mtgrp']]


# In[147]:


# Correlation of grp-M with the target variable and within V columns
corrThresh = 0.65
mtgrp1_high_corr_pairs, mtgrp1_correlationMatrix, mtgrp1_todrop = plotCorrMatrixAndGetColsToDrop(train_transaction, (8,8), corrThresh, m_group_dict['mtgrp'])


# In[148]:


# extract the results from M grp columns

m_retain_cols = list(set(m_group_dict['mtgrp']) - set(mtgrp1_todrop))
m_group_dict_results = {'Columns':m_group_dict['mtgrp'],
               'num columns':len(m_group_dict['mtgrp']),
               'Dropping Columns':mtgrp1_todrop,
               'Num Columns Dropped':len(mtgrp1_todrop),
               'Retain Columns':m_retain_cols,
               'Num Columns Retained': len(m_retain_cols)}
m_group_dict_results


# #### Summary of EDA of M columns

# The M-group contained several categorical column that were label encoded before performing the corerelation analysis. The results of the corerelation analysis yeided 8 columns that were uncorerelated out of the 9 columns present. This groups contained relativeley very less corerelated features and hence we could reatin almost all the features of this group except one.

# ## SELECTED FEATURES FROM THE TRAIN_TRANSACTION DATAFRAME
# 
# ---
# 
# 
# ---
# 
# 
# 
# 
# 

# In[163]:


# Selecting and grouping all the retained features from the V, C, D and M columns

features_to_retain = {'columns_to_retain':V_reduced_dict_results['Retain Columns']+c_group_dict_results['Retain Columns']+\
                      d_group_dict_results['Retain Columns'] + m_group_dict_results['Retain Columns']+\
                      ['TransactionID','isFraud',	'TransactionDT',	'TransactionAmt',	'ProductCD',\
                       'card1',	'card2',	'card3',	'card4',	'card5',	'card6',	'addr1',	'addr2',\
                       'dist1',	'dist2',	'P_emaildomain',	'R_emaildomain'	] }
pd.DataFrame(features_to_retain).to_csv('columns_to_retain.csv')


# In[164]:


len(features_to_retain['columns_to_retain'])


# In[166]:


pd.DataFrame(features_to_retain).to_csv('columns_to_retain.csv')


# In this process, we combine features from various groups: V-reduced (which encompasses features from both fraudulent and complete datasets), as well as the C, D, and M groups, alongside other ungrouped features.
# 
# Following a sequential analysis in our Exploratory Data Analysis (EDA), the total count of features obtained is 119 columns
# 
# These consolidated features are then saved as a .csv file for subsequent steps in building a classification model aimed at accurately predicting fraudulent transactions.

# In[165]:




