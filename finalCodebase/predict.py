import pandas as pd
import tensorflow as tf
import numpy as np



data = pd.read_csv('CorrectDataTesting.csv')
data.iloc[:,2:].head()

numeric_cols = ['Store', 'StoreType', 'DayOfWeek', 
              'Promo', 'StateHoliday', 'Season',
              'Assortment','Promo2','CompetitionOpenSinceYear']

dataCleaned = data[numeric_cols]

dataCleaned = dataCleaned.iloc[:].values
dataCleaned = dataCleaned[:,:-1]
dataCleaned = dataCleaned.reshape((dataCleaned.shape[0], 1, dataCleaned.shape[1]))



model = tf.keras.models.load_model('classifier_hd')
output = model.predict(dataCleaned)
output_denorm = np.exp(output)
kaggeleOutput = data[['Id']].copy()
kaggeleOutput['Sales'] = output_denorm
print(kaggeleOutput)
kaggeleOutput.to_csv('sample_submission_tanh.csv', sep=',', index=False)