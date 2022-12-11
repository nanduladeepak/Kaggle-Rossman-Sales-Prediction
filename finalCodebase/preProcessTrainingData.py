#-------------------------------------
#         Libraries required
#-------------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------
#                   Call csv for prePro
#-------------------------------------------------------------
train_data = pd.read_csv("train.csv", low_memory = False)
store_data = pd.read_csv("store.csv", low_memory = False)

#-------------------------------------------------------------
#                   Functions for prePro
#-------------------------------------------------------------
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

#-------------------------------------------------------------
#           Preprocessing of training data
#-------------------------------------------------------------
# Change format of Date into separate columns: Day, Month and Year
train_data['Year'] = pd.DatetimeIndex(train_data['Date']).year
train_data['Month'] = pd.DatetimeIndex(train_data['Date']).month
train_data['Day'] = pd.DatetimeIndex(train_data['Date']).day
train_data = train_data.drop('Date', axis=1)    # Drop Date column, because it was replaced

# Create column 'Season' using values of column 'Month'
train_data['Season'] = np.where(train_data['Month'].isin([3,4,5]), "Spring",
                 np.where(train_data['Month'].isin([6,7,8]), "Summer",
                 np.where(train_data['Month'].isin([9,10,11]), "Fall",
                 np.where(train_data['Month'].isin([12,1,2]), "Winter", "None"))))

# Find the Outliers of 'Sales' and drop them
High, low = findOutliers(train_data,'Sales')
train_data_outlier = train_data[(train_data['Sales']< High) & (train_data['Sales']> low)]

# Fill blank space of column with mean 
store_data['CompetitionDistance'].fillna(store_data['CompetitionDistance'].mean(), inplace = True)

# Replace missing values of following columns with zeros
replaceWithZeros(store_data,'Promo2SinceWeek')
replaceWithZeros(store_data,'Promo2SinceYear')
replaceWithZeros(store_data,'PromoInterval')
replaceWithZeros(store_data,'CompetitionOpenSinceMonth')
replaceWithZeros(store_data,'CompetitionOpenSinceYear')

# Merged data from testing and store, merged based on Store column
mergedData = pd.merge(train_data, store_data, how='left', on='Store')

# Label encoding of the three columns, because of categorical values
mergedData = LevelEncoding(mergedData, 'StoreType')
mergedData = LevelEncoding(mergedData,'Assortment')
mergedData = LevelEncoding(mergedData, 'Season')

# Replace categorical values with numerical values 
StateHolidayMaps = {'0':0,'a':1,'b':1,'c':1}
mergedData['StateHoliday'].replace(StateHolidayMaps, inplace = True)
PromoIntervalMaps = {'Jan,Apr,Jul,Oct':1, 'Feb,May,Aug,Nov':2, 'Mar,Jun,Sept,Dec':3}
mergedData['PromoInterval'].replace(PromoIntervalMaps, inplace = True)

# Dropping columns because of Feature Selection...
# ... and drop Customers because is not in Test Data
mergedData = mergedData.drop('Promo2SinceWeek', axis=1)
mergedData = mergedData.drop('Promo2SinceYear', axis=1)
mergedData = mergedData.drop('PromoInterval', axis=1)
mergedData = mergedData.drop('CompetitionOpenSinceMonth', axis=1)
mergedData = mergedData.drop('Customers', axis=1)
mergedData = mergedData.drop('Open', axis=1)