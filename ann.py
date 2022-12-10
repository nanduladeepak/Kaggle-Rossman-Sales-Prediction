import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.enable_eager_execution()
# from keras.models import Sequential
# from keras.layers import Dense
# from tensorflow.keras.layers import Activation, Dense
# from tensorflow.keras.optimizers import adam
# from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f'Number of gpus available {len(physical_devices)}')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0],True)


data = pd.read_csv('CorrectDataTraining.csv')
data.iloc[:,2:].head()

numeric_cols = ['Store', 'StoreType', 'DayOfWeek', 
              'Promo', 'StateHoliday', 'Season',
              'Assortment','Promo2']

dataCleaned = data[numeric_cols]
# dataCleaned['CompetitionDistance'] = (dataCleaned['CompetitionDistance']-dataCleaned['CompetitionDistance'].mean())/dataCleaned['CompetitionDistance'].std()
# dataCleaned = (dataCleaned-dataCleaned.mean())/dataCleaned.std()
X_train, X_test, y_train, y_test = train_test_split(dataCleaned, data['Sales'], test_size=0.2, random_state=1000)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1000)



model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=164, input_shape = (8,),activation='relu'),
    tf.keras.layers.Dense(units=300,activation='relu'),
    tf.keras.layers.Dense(units=200,activation='relu'),
    tf.keras.layers.Dense(units=20,activation='relu'),
    tf.keras.layers.Dense(units = 1,activation='relu')
])


model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.Accuracy()])

print(X_train)

model.fit(X_train, y_train, epochs = 15, validation_data = (X_test, y_test))

result = model.predict(X_test)
y_test.columns = ["Actual Sales","Predicted Sales"]

print(y_test.head(10))
print(result[:10])
model.save('classifier_hd')

explainer = shap.DeepExplainer(model,X_train[1:1000].to_numpy())
shap_values = explainer.shap_values(X_test[1:5].to_numpy())
print(shap_values[0][0])
print(explainer.expected_value)
shap.summary_plot(shap_values, plot_type = 'bar', feature_names = X_test.columns.tolist())


fig = plt.figure(1)
plt.plot(range(0,100), y_test.head(100)) # plot first line
plt.plot(range(0,100), result[:100]) # plot second line
plt.show()

