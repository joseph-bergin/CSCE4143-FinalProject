import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, accuracy_score, log_loss
import psutil
import warnings

warnings.filterwarnings("ignore")

def print_memory_usage():
    pid = psutil.Process()
    memory_info = pid.memory_info()
    print(f"Memory used: {memory_info.rss / (1024 * 1024): .2f} MB")

# Load the full dataset
train_df = pd.read_csv('train.csv')

# Drop rows with latitude greater than 38
train_df = train_df[train_df['Y'] <= 38]
train_df = train_df.drop(['Dates', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address'], axis=1)
train_df = train_df.rename(columns={'X': 'Longitude', 'Y': 'Latitude'})

# Encode the 'Category' column
label_encoder = LabelEncoder()
train_df['Category'] = label_encoder.fit_transform(train_df['Category'])

# Split the subset into features (X) and target variable (y)
X = train_df[['Longitude', 'Latitude']]
y = train_df['Category']

# Print the memory usage before training the Naive Bayes model
print_memory_usage()

# Split the subset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Fit the classifier to the training data
nb_classifier.fit(X_train, y_train)

# Print the memory usage after training the Naive Bayes model
print_memory_usage()

# Make predictions on the test set (get predicted probabilities)
y_pred_proba = nb_classifier.predict_proba(X_test)
y_pred = nb_classifier.predict(X_test)

# Calculate log loss
logloss = log_loss(y_test, y_pred_proba)

# Calculate accuracy and precision scores
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

# Print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Log Loss:", logloss)