import pandas as pd
import numpy as np

data = pd.read_csv('C:/Users/Nithan/AndroidStudioProjects/HomePredictorOne/python_pred_sect/dataset/home_data.txt')  
print(data.head())

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X,y = data.values[:,0:3], data.values[:,3] 
# For X values, we take the first two columns, and for y values, we take the third column

print(X)
print(y)


# Build model and create train and test sets
model = LinearRegression()
model.fit(X,y)




pred = model.predict([[1500,1,3]])[0]  # Predicting the price of a house with 2000 square feet and 3 rooms
#predict the price of a house with 3 rooms and 2000 square feet
print(f"Predict the price of the house with 3 rooms and 2000 square feet and zone 2: ${pred:.2f}")  # Display the predicted price rounded to 2 decimal places}")


from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = {
    ('input_area_and_rooms_and_zone', FloatTensorType([None, 3]))  # Input type for the model where the first column is area, the second column is number of rooms, and the third column is zone
}
# Input type for the model where the first column is area and the second column is number of rooms and the third column is zone
# Convert the model to ONNX format
converted_model = convert_sklearn(model, initial_types=initial_type)

with open("house_price_model.onnx", "wb") as f:
    f.write(converted_model.SerializeToString())  # Save the ONNX model to a file

# Integrate this onxx model into your mobile app
# You can use libraries like ONNX Runtime or TensorFlow Lite to run the model on mobile devices
# For example, in Android, you can use ONNX Runtime for Android to load and run the model
