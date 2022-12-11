import pandas as pd
import tensorflow as tf
import numpy as np

data = pd.read_csv('CorrectDataTraining.csv')
from sklearn.model_selection import train_test_split

numeric_cols = ['Store', 'StoreType', 'DayOfWeek', 
              'Promo', 'StateHoliday', 'Season',
              'Assortment','Promo2','Sales']

X_train, X_test, y_train, y_test = train_test_split(data[numeric_cols], data['Sales'], test_size=0.2, random_state=1000)

y_train_array = y_train.iloc[:].values

y_train = y_train_array[:]

trainMin = min(y_train)
trainMax = max(y_train)


data = pd.read_csv('CorrectDataTesting.csv')
data.iloc[:,2:].head()

numeric_cols = ['Store', 'StoreType', 'DayOfWeek', 
              'Promo', 'StateHoliday', 'Season',
              'Assortment','Promo2','CompetitionOpenSinceYear']

dataCleaned = data[numeric_cols]

dataCleaned = dataCleaned.iloc[:].values
dataCleaned = dataCleaned[:,:-1]
dataCleaned = dataCleaned.reshape((dataCleaned.shape[0], 1, dataCleaned.shape[1]))



model = tf.keras.models.load_model('rnnSigmoidMinMaxModel_hd')
output = model.predict(dataCleaned)
output_denorm = (output*(trainMax-trainMin))+trainMin
kaggeleOutput = data[['Id']].copy()
kaggeleOutput['Sales'] = output_denorm
print(kaggeleOutput)
kaggeleOutput.to_csv('sample_submission_sigmoid.csv', sep=',', index=False)