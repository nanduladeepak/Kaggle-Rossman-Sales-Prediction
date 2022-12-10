import pandas as pd
import tensorflow as tf



data = pd.read_csv('CorrectDataTesting.csv')
data.iloc[:,2:].head()

numeric_cols = ['Store', 'StoreType', 'DayOfWeek', 
              'Promo', 'StateHoliday', 'Season',
              'Assortment','Promo2']

dataCleaned = data[numeric_cols]


model = tf.keras.models.load_model('classifier_hd')
output = model.predict(dataCleaned)

kaggeleOutput = data[['Id']].copy()
kaggeleOutput['Sales'] = output
print(kaggeleOutput)
kaggeleOutput.to_csv('sample_submission.csv', sep=',', index=False)