from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.tree import plot_tree


train = pd.read_csv('onehot_sample.csv')
feature_names = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
                 'no_of_week_nights', 'required_car_parking_space',
                 'lead_time', 'arrival_year', 'arrival_month',
                 'arrival_date', 'repeated_guest',
                 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                 'avg_price_per_room', 'no_of_special_requests',
                 'meal_type_1', 'meal_type_2', 'meal_type_3', 'meal_type_4',
                 'room_type_1', 'room_type_2', 'room_type_3', 'room_type_4',
                 'room_type_5', 'room_type_6', 'room_type_7', 'Corporate',
                 'Complementary', 'Online', 'Offline', 'Aviation']

# Decision Tree 학습/예측/평가
X = train[feature_names]
y = train["booking_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(
    max_depth=20, min_samples_leaf=6, min_samples_split=6, criterion="gini", max_features=25)
scores = cross_val_score(dt_model, X, y, cv=10)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

params = {
    'max_depth': [8, 10, 15, 20],
    'min_samples_leaf': [2, 4, 6, 8],
    'min_samples_split': [2, 4, 6, 8],
    'max_features': [25, 30, 25, 40, 'sqrt'],
    'max_leaf_nodes': [15, 20, 25, 30, 35]
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=dt_model,
                           param_grid=params,
                           cv=10, n_jobs=-1, verbose=1, scoring="accuracy")

grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)
score_df.nlargest(5, "mean_test_score")

dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))

real_test = pd.read_csv('onehot_test.csv')

feature_names = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
                 'no_of_week_nights', 'required_car_parking_space',
                 'lead_time', 'arrival_year', 'arrival_month',
                 'arrival_date', 'repeated_guest',
                 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                 'avg_price_per_room', 'no_of_special_requests',
                 'meal_type_1', 'meal_type_2', 'meal_type_3', 'meal_type_4',
                 'room_type_1', 'room_type_2', 'room_type_3', 'room_type_4',
                 'room_type_5', 'room_type_6', 'room_type_7', 'Corporate',
                 'Complementary', 'Online', 'Offline', 'Aviation']

test = real_test[feature_names]
y_pred = dt_model.predict(test)
print(y_pred)
sample = pd.DataFrame()
sample['Booking_ID'] = real_test['Booking_ID']
sample['booking_status'] = y_pred
sample.to_csv('sample_kingwhangjang.csv', index=False)
real_test.keys()
