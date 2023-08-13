import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

# Step 1: Data Preparation
data = pd.read_excel("D:\Fwd-data.xlsx")
X = data.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13]].values
y = data.iloc[:, [4, 5, 6]].values

# Step 2: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Data Normalization
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Neural Network Training
input_size = X_train.shape[1]
output_size = y_train.shape[1]
hidden_units = 10

# Define the neural network model
model = MLPRegressor(hidden_layer_sizes=(hidden_units,), activation='logistic', solver='lbfgs')

# Train the model
model.fit(X_train, y_train)

# User Input
CBR_value = float(input("Enter CBR Value: "))
MSA_value = float(input("Enter MSA Value: "))
bituminous_thickness = float(input("Enter Bituminous Thickness: "))
granular_thickness = float(input("Enter Granular Thickness: "))
vertical_displacement_at_0mm = float(input("Enter Vertical Displacement at 0mm RC: "))
vertical_displacement_at_20mm = float(input("Enter Vertical Displacement at 20mm RC: "))
vertical_displacement_at_30mm = float(input("Enter Vertical Displacement at 30mm RC: "))
vertical_displacement_at_45mm = float(input("Enter Vertical Displacement at 45mm RC: "))
vertical_displacement_at_60mm = float(input("Enter Vertical Displacement at 60mm RC: "))
vertical_displacement_at_90mm = float(input("Enter Vertical Displacement at 90mm RC: "))
vertical_displacement_at_150mm = float(input("Enter Vertical Displacement at 150mm RC: "))

# Normalize the user input data
user_input = np.array([[CBR_value, MSA_value, bituminous_thickness, granular_thickness,
                        vertical_displacement_at_0mm, vertical_displacement_at_20mm,
                        vertical_displacement_at_30mm, vertical_displacement_at_45mm,
                        vertical_displacement_at_60mm, vertical_displacement_at_90mm,
                        vertical_displacement_at_150mm]])
user_input_scaled = scaler.transform(user_input)

# Predict the output using the trained model
output = model.predict(user_input_scaled)

# Display the predicted output in a table format
output_df = pd.DataFrame({
    "Bituminous Modulus": [output[0][0]],
    "Granular Modulus": [output[0][1]],
    "Sub-base Modulus": [output[0][2]]
}, index=["Predicted Output"])

print(output_df)
