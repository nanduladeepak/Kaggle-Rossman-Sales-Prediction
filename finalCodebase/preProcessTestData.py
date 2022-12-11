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


def LevelEncoding(df,col_name):
    labelEncoder = preprocessing.LabelEncoder()
    df[col_name] = labelEncoder.fit_transform(df[col_name])
    return df

def Percentage(df,colname,val):
    return round(df[df[colname]==val].shape[0]/df.shape[0] * 100,2)

def replaceWithZeros(df,col_name):
    df[col_name].fillna(0,inplace =True)
    return df

def findOutliers(df,col_name):
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    high_cutoff = Q3 + 1.5*IQR
    low_cutoff = Q1 - 1.5*IQR
    return high_cutoff, low_cutoff

def LevelEncoding(df,col_name):
    labelEncoder = preprocessing.LabelEncoder()
    df[col_name] = labelEncoder.fit_transform(df[col_name])
    return df


data = pd.read_csv('DataSet/test.csv')

data['Year'] = pd.DatetimeIndex(data['Date']).year
data['Month'] = pd.DatetimeIndex(data['Date']).month
data['Day'] = pd.DatetimeIndex(data['Date']).day
data = data.drop('Date', axis=1)

store_data = pd.read_csv("DataSet/store.csv", low_memory = False)

store_data['CompetitionDistance'].fillna(store_data['CompetitionDistance'].mean(), inplace = True)
replaceWithZeros(store_data,'Promo2SinceWeek')
replaceWithZeros(store_data,'Promo2SinceYear')
replaceWithZeros(store_data,'PromoInterval')
replaceWithZeros(store_data,'CompetitionOpenSinceMonth')
replaceWithZeros(store_data,'CompetitionOpenSinceYear')

mergedData = pd.merge(data, store_data, how='left', on='Store')


mergedData = LevelEncoding(mergedData, 'StoreType')
mergedData = LevelEncoding(mergedData,'Assortment')
# mergedData = LevelEncoding(mergedData, 'Season')
StateHolidayMaps = {'0':0,'a':1,'b':1,'c':1}
mergedData['StateHoliday'].replace(StateHolidayMaps, inplace = True)
PromoIntervalMaps = {'Jan,Apr,Jul,Oct':1, 'Feb,May,Aug,Nov':2, 'Mar,Jun,Sept,Dec':3}
mergedData['PromoInterval'].replace(PromoIntervalMaps, inplace = True)

mergedData = mergedData.drop('Promo2SinceWeek', axis=1)
mergedData = mergedData.drop('Promo2SinceYear', axis=1)
mergedData = mergedData.drop('PromoInterval', axis=1)

mergedData = mergedData.drop('CompetitionOpenSinceMonth', axis=1)


mergedData['Season'] = np.where(mergedData['Month'].isin([3,4,5]), "Spring",
                 np.where(mergedData['Month'].isin([6,7,8]), "Summer",
                 np.where(mergedData['Month'].isin([9,10,11]), "Fall",
                 np.where(mergedData['Month'].isin([12,1,2]), "Winter", "None"))))
mergedData = LevelEncoding(mergedData, 'Season')

# Save dataset to use in testing
mergedData.to_csv('CorrectDataTesting.csv')