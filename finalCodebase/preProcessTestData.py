#------------------------------------
#         Libraries Required
#------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn import preprocessing
import sklearn
import seaborn as sns
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
sns.set_style("darkgrid")
import plotly.figure_factory as ff
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

#------------------------------------
#        Functions for prePro
#------------------------------------
def LabelEncoding(df,col_name):     # Label Encoding Function
    labelEncoder = preprocessing.LabelEncoder()
    df[col_name] = labelEncoder.fit_transform(df[col_name])
    return df

def Percentage(df,colname,val):     # Returns percentage of value in column 
    return round(df[df[colname]==val].shape[0]/df.shape[0] * 100,2)

def replaceWithZeros(df,col_name):  # Replace column with zeros function
    df[col_name].fillna(0,inplace =True)
    return df

def findOutliers(df,col_name):      # Finding outliers in columns
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    high_cutoff = Q3 + 1.5*IQR
    low_cutoff = Q1 - 1.5*IQR
    return high_cutoff, low_cutoff

# Read testing data to preProcess
data = pd.read_csv('test.csv')

# Change format of Date into separate columns: Day, Month and Year
data['Year'] = pd.DatetimeIndex(data['Date']).year
data['Month'] = pd.DatetimeIndex(data['Date']).month
data['Day'] = pd.DatetimeIndex(data['Date']).day
data = data.drop('Date', axis=1)    # Drop Date column, because it was replaced

# Read store data to merge with testing data
store_data = pd.read_csv("store.csv", low_memory = False)

# Fill blank space of column with mean 
store_data['CompetitionDistance'].fillna(store_data['CompetitionDistance'].mean(), inplace = True)

# Replace missing values of following columns with zeros
replaceWithZeros(store_data,'Promo2SinceWeek')
replaceWithZeros(store_data,'Promo2SinceYear')
replaceWithZeros(store_data,'PromoInterval')
replaceWithZeros(store_data,'CompetitionOpenSinceMonth')
replaceWithZeros(store_data,'CompetitionOpenSinceYear')

# Merged data from testing and store, merged based on Store column
mergedData = pd.merge(data, store_data, how='left', on='Store')

# Label encoding of the two columns, because of categorical values
mergedData = LabelEncoding(mergedData, 'StoreType')
mergedData = LabelEncoding(mergedData,'Assortment')

# Replace categorical values with numerical values 
StateHolidayMaps = {'0':0,'a':1,'b':1,'c':1}
mergedData['StateHoliday'].replace(StateHolidayMaps, inplace = True)
PromoIntervalMaps = {'Jan,Apr,Jul,Oct':1, 'Feb,May,Aug,Nov':2, 'Mar,Jun,Sept,Dec':3}
mergedData['PromoInterval'].replace(PromoIntervalMaps, inplace = True)

# Dropping columns because of Feature Selection in Training
mergedData = mergedData.drop('Promo2SinceWeek', axis=1)
mergedData = mergedData.drop('Promo2SinceYear', axis=1)
mergedData = mergedData.drop('PromoInterval', axis=1)
mergedData = mergedData.drop('CompetitionOpenSinceMonth', axis=1)

# Create Season column in merged data because it was selected in SFS
mergedData['Season'] = np.where(mergedData['Month'].isin([3,4,5]), "Spring",
                 np.where(mergedData['Month'].isin([6,7,8]), "Summer",
                 np.where(mergedData['Month'].isin([9,10,11]), "Fall",
                 np.where(mergedData['Month'].isin([12,1,2]), "Winter", "None"))))
mergedData = LabelEncoding(mergedData, 'Season')

# Save dataset to use in testing
mergedData.to_csv('CorrectDataTesting.csv')