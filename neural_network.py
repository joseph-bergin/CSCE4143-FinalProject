import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyxlsb import open_workbook

# Loading file_path
# Replace 'your_file.xlsb' with the actual file path
file_path = 'train.xlsb'

# Open the workbook
with open_workbook(file_path) as wb:
    # Select the first sheet
    with wb.get_sheet(1) as sheet:
        # Read data from the sheet
        data = [sheet.row(row) for row in sheet.rows()]

# Display the first few rows of the data
print(data[:5])


# Assume you have your features (X) and labels (y) for training and testing data


# X_train, X_test, y_train, y_test = load_data()

# Split your data into training and testing sets
# Adjust the test_size and random_state parameters as needed
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)  # Convert probabilities to class labels

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy on test data: {accuracy * 100:.2f}%")
