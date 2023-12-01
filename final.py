import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


#-------------- Import datasets --------------#
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
train_data = train_data.drop(["Descript", "Resolution"], axis=1)
#----------------------------------------------#


#-------------- Clean data / Feature engineering --------------#
day_map = {
    'Sunday': 0,
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6
}
train_data['DayOfWeek'] = train_data['DayOfWeek'].map(day_map)
test_data['DayOfWeek'] = test_data['DayOfWeek'].map(day_map)

train_data.loc[(train_data['X'] == -120.5) | (train_data['Y'] == 90.0), ['X', 'Y']] = None
test_data.loc[(test_data['X'] == -120.5) | (test_data['Y'] == 90.0), ['X', 'Y']] = None

imputer = SimpleImputer(strategy='mean')
train_data[['X', 'Y']] = imputer.fit_transform(train_data[['X', 'Y']])
test_data[['X', 'Y']] = imputer.transform(test_data[['X', 'Y']])

train_data['X_rotated_30'] = train_data['X'] * 0.866 - train_data['Y'] * 0.5
train_data['Y_rotated_30'] = train_data['X'] * 0.5 + train_data['Y'] * 0.866

train_data['X_rotated_45'] = train_data['X'] * 0.707 + train_data['Y'] * 0.707
train_data['Y_rotated_45'] = -train_data['X'] * 0.707 + train_data['Y'] * 0.707

train_data['X_rotated_60'] = train_data['X'] * 0.5 + train_data['Y'] * 0.866
train_data['Y_rotated_60'] = -train_data['X'] * 0.866 + train_data['Y'] * 0.5

train_data['R'] = np.sqrt(train_data['X']**2 + train_data['Y']**2)
train_data['Theta'] = np.arctan2(train_data['Y'], train_data['X'])

test_data['X_rotated_30'] = train_data['X'] * 0.866 - train_data['Y'] * 0.5
test_data['Y_rotated_30'] = train_data['X'] * 0.5 + train_data['Y'] * 0.866

test_data['X_rotated_45'] = train_data['X'] * 0.707 + train_data['Y'] * 0.707
test_data['Y_rotated_45'] = -train_data['X'] * 0.707 + train_data['Y'] * 0.707

test_data['X_rotated_60'] = train_data['X'] * 0.5 + train_data['Y'] * 0.866
test_data['Y_rotated_60'] = -train_data['X'] * 0.866 + train_data['Y'] * 0.5

test_data['R'] = np.sqrt(train_data['X']**2 + train_data['Y']**2)
test_data['Theta'] = np.arctan2(train_data['Y'], train_data['X'])
#---------------------------------------------------------------------------#


#-------------- Split data into features / Target / Train / Validation -----#
X = train_data[['Y', 'X', 'DayOfWeek', 'PdDistrict', 'X_rotated_30', 'Y_rotated_30', 'X_rotated_45', 'Y_rotated_45', 'X_rotated_60', 'Y_rotated_60', 'R', 'Theta']]
y = train_data['Category']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
#---------------------------------------------------------------------------#


#-------------- More data cleaning --------------#
numeric_features = ['Y', 'X', 'DayOfWeek', 'X_rotated_30', 'Y_rotated_30', 'X_rotated_45', 'Y_rotated_45', 'X_rotated_60', 'Y_rotated_60', 'R', 'Theta']
categorical_features = ['PdDistrict']


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
#------------------------------------------------#


#-------------- Trains a linear regression model --------------#
def run_lr():
    lr_model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', LogisticRegression(random_state=42, max_iter=1000))])
    lr_model.fit(X_train, y_train)
    lr_val_pred = lr_model.predict_proba(X_val)

    logloss = log_loss(y_val, lr_val_pred)
    print(f"Log Loss on Validation Set: {logloss}")
    lr_model.fit(X, y)
    lr_pred = lr_model.predict_proba(test_data[['Y', 'X', 'DayOfWeek', 'X_rotated_30', 'Y_rotated_30', 'X_rotated_45', 'Y_rotated_45', 'X_rotated_60', 'Y_rotated_60', 'R', 'Theta']])
    
    class_names = lr_model.named_steps['classifier'].classes_
    lr_df = pd.DataFrame(lr_pred, columns=class_names)
    lr_df.insert(0, 'Incident_ID', test_data.index)

    lr_df.to_csv('logistic_regression.csv', index=False)
    return
#--------------------------------------------------------------#


#-------------- Trains a K-Nearest Neighbors model --------------#
def run_knn():
    knn_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=85))  
    ])

    knn_model.fit(X_train, y_train)
    knn_val_pred = knn_model.predict_proba(X_val)

    logloss = log_loss(y_val, knn_val_pred)
    print(f"Log Loss on Validation Set: {logloss}")
    knn_model.fit(X, y)
    knn_pred = knn_model.predict_proba(test_data[['Y', 'X', 'DayOfWeek', 'PdDistrict', 'X_rotated_30', 'Y_rotated_30', 'X_rotated_45', 'Y_rotated_45', 'X_rotated_60', 'Y_rotated_60', 'R', 'Theta']])

    class_names = knn_model.named_steps['classifier'].classes_
    knn_df = pd.DataFrame(knn_pred, columns=class_names)
    knn_df.insert(0, 'Incident_ID', test_data.index)

    knn_df.to_csv('knn.csv', index=False)
    return
#----------------------------------------------------------------#


#-------------- Trains a gradient boost model --------------#
def run_grad_boost():
    gb_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', GradientBoostingClassifier(n_estimators=20,learning_rate = 0.2,
                      max_depth = 11))])

    gb_model.fit(X_train, y_train)
    gb_val_pred = gb_model.predict_proba(X_val)

    logloss = log_loss(y_val, gb_val_pred)
    print(f"Log Loss on Validation Set (Gradient Boosting): {logloss}")

    gb_pred = gb_model.predict_proba(test_data[['Y', 'X', 'DayOfWeek', 'PdDistrict', 'X_rotated_30', 'Y_rotated_30', 'X_rotated_45', 'Y_rotated_45', 'X_rotated_60', 'Y_rotated_60', 'R', 'Theta']])

    class_names_gb = gb_model.named_steps['classifier'].classes_
    gb_df = pd.DataFrame(gb_pred, columns=class_names_gb)
    gb_df.insert(0, 'Incident_ID', test_data.index)

    gb_df.to_csv('gb.csv', index=False)
    return
#----------------------------------------------------------------#


#-------------- Trains a random forest classifier model --------------#
def run_random_forest():
    rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=18, random_state=42))])

    rf_model.fit(X_train, y_train)
    rf_val_pred = rf_model.predict_proba(X_val)

    logloss = log_loss(y_val, rf_val_pred)
    print(f"Log Loss on Validation Set (Random Forest): {logloss}")
    

    rf_pred = rf_model.predict_proba(test_data[['Y', 'X', 'DayOfWeek', 'PdDistrict', 'X_rotated_30', 'Y_rotated_30', 'X_rotated_45', 'Y_rotated_45', 'X_rotated_60', 'Y_rotated_60', 'R', 'Theta']])

    class_names = rf_model.named_steps['classifier'].classes_
    rf_df = pd.DataFrame(rf_pred, columns=class_names)
    rf_df.insert(0, 'Incident_ID', test_data.index)

    rf_df.to_csv('random_forest.csv', index=False)
    return
#--------------------------------------------------------------------#


#------------ Main Program ---------------#
# run_random_forest()
# run_knn()
# run_lr()
# run_grad_boost()
#-----------------------------------------#
