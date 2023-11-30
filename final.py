from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action="ignore", category= FutureWarning)

# Load your training and test datasets (replace 'train.csv' and 'test.csv' with your actual file paths)
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_df = train_df.rename(columns={'X': 'Longitude', 'Y': 'Latitude'})
test_df = test_df.rename(columns={'X': 'Longitude', 'Y': 'Latitude'})

train_df.drop('Descript', axis=1, inplace=True)
test_df.drop('Resolution',axis=1,inplace=True)

model = KMeans(n_clusters=4, random_state=42)
model.fit(train_df)
# Print the clusters assigned to locations in the test dataset
# print(test_df[['Latitude', 'Longitude', 'Cluster']])


# def optimize(data, max_k):
#     means = []
#     inertias = []
#     for k in range(1,max_k):
#         kmeans = KMeans(n_clusters=k)
#         kmeans.fit(data)
#         means.append(k)
#         inertias.append(kmeans.inertia_)
#     fig = plt.subplots(figsize=(10,5))
#     plt.plot(means, inertias, 'o-')
#     plt.xlabel('Number of Clusters')
#     plt.ylabel('Inertia')
#     plt.grid(True)
#     plt.show()
# warnings.simplefilter(action="ignore", category=FutureWarning)
# col_names = ['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']
# df = pd.read_csv("train.csv", header = 0, names = ['Dates', 'Category', 'Description', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'Longitude', 'Latitude'])
# features = df[['Longitude', 'Latitude','DayOfWeek', 'Category']]
# df.drop(df.columns[[2,5]], axis=1, inplace=True)
# #optimize(df[['Longitude', 'Latitude']], 10)
# le = LabelEncoder()
# categorical_columns = ['DayOfWeek', 'PdDistrict', 'Address', 'Category']
# for column in categorical_columns:
#     df[column] = le.fit_transform(df[column])
# df.drop('Dates', axis=1, inplace=True)
# X = df.drop('Category', axis=1)
# latitude_condition = X['Latitude'] > 38
# X.drop(X[latitude_condition].index, inplace=True)

# longitude_condition1 = X['Longitude'] > -122
# X.drop(X[longitude_condition1].index, inplace=True)
# y = df['Category']

# cluster_features = ['Longitude', 'Latitude']
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X[cluster_features])
# num_clusters = 3
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# X['cluster'] = kmeans.fit_predict(X_scaled)
# X['Category'] = y
# cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
# X['crime_prob'] = kmeans.predict(X_scaled)
# print("Cluster Centers:")
# print(pd.DataFrame(cluster_centers, columns=cluster_features))
# feature1 = 'Longitude'
# feature2 = 'Latitude'
# # Display the distribution of crime categories within each cluster
# cluster_analysis = X.groupby(['cluster', 'Category']).size().unstack(fill_value=0)
# print("\nCluster Analysis:")
# print(cluster_analysis)
# cluster_probabilities = X.groupby(['cluster', 'Category']).size() / X.groupby('cluster').size()
# cluster_probabilities = cluster_probabilities.unstack(fill_value=0)

# for cluster_id in cluster_probabilities.index:
#     print(f"\nCluster {cluster_id}:")

#     for crime_category in cluster_probabilities.columns:
#         probability = cluster_probabilities.loc[cluster_id, crime_category]
#         print(f"{crime_category}: {probability:.2%}")

# overall_probabilities = X['Category'].value_counts(normalize=True)

# # Print the overall probabilities for each crime category
# print("Overall Crime Category Probabilities:")
# for crime_category, probability in overall_probabilities.items():
#     print(f"{crime_category}: {probability:.2%}")
# print(X[['cluster', 'Category']].head())
