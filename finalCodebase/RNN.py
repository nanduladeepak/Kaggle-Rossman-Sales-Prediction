#------------------------------------
#         Libraries Required
#------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#--------------------------------------------------------
#            Arrangement of data for model
#--------------------------------------------------------
data = pd.read_csv('CorrectDataTraining.csv')           # Read .csv with training data preprocessed correctly

numeric_cols = ['Store', 'StoreType', 'DayOfWeek',      # Feature selected to do RNN model
              'Promo', 'StateHoliday', 'Season',
              'Assortment','Promo2','Sales']

#Splitting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(data[numeric_cols], data['Sales'], test_size=0.2, random_state=1000)

# Re-shaping the data
X_train_array = X_train.iloc[:,:].values
X_test_array =  X_test.iloc[:,:].values
y_train_array = y_train.iloc[:].values
y_test_array =  y_test.iloc[:].values

# Reseting the variables
X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []

# Separating final column to use as output value, and  dropping it in input value
X_train, y_train = X_train_array[:,:-1], y_train_array[:]
X_test, y_test = X_test_array[:,:-1], y_test_array[:]

y_test = np.log(y_test)     # Normalizing output with log
y_train = np.log(y_train)   # Normalizing output with log

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # Re-shaping for use in RNN mode
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))      # Re-shaping for use in RNN mode

#------------------------------------------------------
#               Creation of RNN Model
#------------------------------------------------------

model = tf.keras.Sequential()   # Sequential model

# Adding the first LSTM layer and some Dropout regularisation
model.add(tf.keras.layers.LSTM(units = 200,return_sequences=True, input_shape = (1,8),activation='relu'))
model.add(tf.keras.layers.Dropout(0.2)) # 20% Dropout

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(tf.keras.layers.LSTM(units = 200,return_sequences=True,activation='relu'))
model.add(tf.keras.layers.Dropout(0.2)) # 20% Dropout

# Adding the output layer
model.add(tf.keras.layers.LSTM(units = 50,activation='relu'))
model.add(tf.keras.layers.Dropout(0.2)) # 20% Dropout
model.add(tf.keras.layers.Dense(1,activation='relu'))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.Accuracy()])

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 12,batch_size=512, validation_data = (X_test, y_test))
model.save('rnnModel_hd')

# Model prediction with testing data
result = model.predict(X_test)

# Denormalizing values to see actual predictions
result_denorm = np.exp(result)
y_test_denorm = np.exp(y_test)
df = pd.DataFrame(y_test_denorm)

# Saving model to use in testing data provided in kaggle
model.save('classifier_hd')