import pandas as pd
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from keras.layers import Dropout
import matplotlib.pyplot as plt

data = pd.read_csv('CorrectDataTraining.csv')
data.iloc[:,2:].head()

from sklearn.model_selection import train_test_split

numeric_cols = ['Store', 'StoreType', 'DayOfWeek', 
              'Promo', 'StateHoliday', 'Season',
              'Assortment','Promo2','Sales']

#splitting data into train test data
X_train, X_test, y_train, y_test = train_test_split(data[numeric_cols], data['Sales'], test_size=0.2, random_state=1000)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1000)


X_train_array = X_train.iloc[:,:].values
X_test_array =  X_test.iloc[:,:].values
y_train_array = y_train.iloc[:].values
y_test_array =  y_test.iloc[:].values

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []

X_train, y_train = X_train_array[:,:-1], y_train_array[:]
X_test, y_test = X_test_array[:,:-1], y_test_array[:]

print(X_train.shape, y_train.shape,  X_test.shape, y_test.shape)
y_test = np.log(y_test)
y_train = np.log(y_train)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))



model = tf.keras.Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(tf.keras.layers.LSTM(units = 200,return_sequences=True, input_shape = (1,8),activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(tf.keras.layers.LSTM(units = 200,return_sequences=True,activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

# Adding the output layer

model.add(tf.keras.layers.LSTM(units = 50,activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(1,activation='relu'))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.Accuracy()])



# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 12,batch_size=512, validation_data = (X_test, y_test))
model.save('rnnModel_hd')


result = model.predict(X_test)
result_denorm = np.exp(result)
y_test_denorm = np.exp(y_test)
df = pd.DataFrame(y_test_denorm)


model.save('classifier_hd')

fig = plt.figure(1)
plt.plot(range(0,100), y_test_denorm[:100]) # plot first line
plt.plot(range(0,100), result_denorm[:100]) # plot second line
plt.show()

