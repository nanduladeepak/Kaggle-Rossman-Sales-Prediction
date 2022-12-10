import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
data = pd.read_csv('trainMergeReady.csv')
data.iloc[:,2:].head()
from sklearn.model_selection import train_test_split

numeric_cols = ['Store', 'DayOfWeek', 
               'Customers', 'Open', 'Promo', 'StateHoliday',
              'SchoolHoliday', 'StoreType', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 
              'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

X_train, X_test, y_train, y_test = train_test_split(data[numeric_cols], data['Sales'], test_size=0.2, random_state=1000)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1000)
y_test.head(10)
plt.plot(range(0,10),y_test.head(10))
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (15, 1)))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(32))
model.add(Dropout(0.2))

model.add(Dense(1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=["mean_absolute_error"])

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 2, validation_data = (X_test, y_test))
result = model.predict(X_test)
y_test.columns = ["Actual Sales","Predicted Sales"]
print(y_test.head(10))
print(result[:10])
fig = plt.figure(1)
plt.plot(range(0,100), y_test.head(100)) # plot first line
plt.plot(range(0,100), result[:100]) # plot second line
plt.show()