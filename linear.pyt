import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data
df = pd.read_csv("insurance.csv")  # Replace with the actual dataset path

# Convert categorical data to numerical
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Split data into training and testing sets (80%-20%)
train_dataset, test_dataset = train_test_split(df, test_size=0.2, random_state=42)

# Separate labels (expenses)
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# Normalize numerical features
scaler = StandardScaler()
train_dataset = scaler.fit_transform(train_dataset)
test_dataset = scaler.transform(test_dataset)

# Build the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(train_dataset.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)  # Single output neuron for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# Train the model
model.fit(train_dataset, train_labels, epochs=100, validation_split=0.2, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(test_dataset, test_
