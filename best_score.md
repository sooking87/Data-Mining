```python
# 87 프로 피쳐
# train set = onehot_sample.csv
# feature_names = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
#                  'no_of_week_nights', 'required_car_parking_space',
#                  'lead_time', 'arrival_year', 'arrival_month',
#                  'arrival_date', 'repeated_guest',
#                  'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
#                  'avg_price_per_room', 'no_of_special_requests',
#                  'meal_type_1', 'meal_type_2', 'meal_type_3', 'meal_type_4',
#                  'room_type_1', 'room_type_2', 'room_type_3', 'room_type_4',
#                  'room_type_5', 'room_type_6', 'room_type_7', 'Corporate',
#                  'Complementary', 'Online', 'Offline', 'Aviation']
```

```python
0.87825

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sb


# 목표 -> 모든 숫자를 정규화, 차원 최소화
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# cate -> num: canceled: 1, not canceled: 0
booking_status = {
    "Canceled": 1,
    "Not_Canceled": 0
}
train['booking_status'] = train['booking_status'].map(booking_status)
train = train.drop(columns=['Booking_ID'])

# pairplot을 통해서 의미 없음을 발견
train = train.drop(["arrival_year", "arrival_month", "arrival_date"], axis=1)
test = test.drop(["arrival_year", "arrival_month", "arrival_date"], axis=1)

# df[col].value_counts() 를 통해서 나온 최빈 순서대로 번호를 메김
meal_plan = {"Meal Plan 1": 0, "Not Selected": 1,
             "Meal Plan 2": 2, "Meal Plan 3": 3}
room_type = {
    "Room_Type 1": 0,
    "Room_Type 4": 1,
    "Room_Type 6": 3,
    "Room_Type 2": 2,
    "Room_Type 5": 4,
    "Room_Type 7": 5,
    "Room_Type 3": 6
}
market_segment = {
    "Online": 1,
    "Offline": 0,
    "Corporate": 2,
    "Complementary": 4,
    "Aviation": 3
}
# cate to num
train['type_of_meal_plan'] = train['type_of_meal_plan'].map(meal_plan)
train['room_type_reserved'] = train['room_type_reserved'].map(room_type)
train['market_segment_type'] = train['market_segment_type'].map(market_segment)

test['type_of_meal_plan'] = test['type_of_meal_plan'].map(meal_plan)
test['room_type_reserved'] = test['room_type_reserved'].map(room_type)
test['market_segment_type'] = test['market_segment_type'].map(market_segment)

# train feature creation
train['nights'] = train['no_of_weekend_nights'] + train['no_of_week_nights']
train['lead_x_nights'] = train['lead_time'] * train['nights']

train['no_of_adults'] += train['no_of_children']
train = train.drop(['no_of_children'], axis=1)

# test feature creation
test['nights'] = test['no_of_weekend_nights'] + test['no_of_week_nights']
test['lead_x_nights'] = test['lead_time'] * test['nights']

test['no_of_adults'] += test['no_of_children']
test = test.drop(['no_of_children'], axis=1)

# MinMaxScaler
features = [
    'no_of_adults',
    'no_of_weekend_nights',
    'no_of_week_nights',
    'lead_time',
    'no_of_previous_cancellations',
    'no_of_previous_bookings_not_canceled',
    'avg_price_per_room',
    'no_of_special_requests',
    'lead_x_nights',
    'nights'
]
scaler = MinMaxScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

# train.to_csv('real_real_final_train.csv', index=False)
# test.to_csv('real_real_final_test.csv', index=False)

# 모델링
feature_names = ['no_of_adults',
                 'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved',
                 'lead_time', 'market_segment_type',
                 'avg_price_per_room', 'no_of_special_requests',
                 'nights', 'lead_x_nights']

X = train[feature_names]
y = train["booking_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

LogisticRegression 학습/예측/평가
knn_model = KNeighborsClassifier(
    metric='euclidean', n_neighbors=29, weights='distance', algorithm='kd_tree', leaf_size=5)

scores = cross_val_score(knn_model, X, y, cv=10)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
print('NearestNeighbors 정확도: {0:.4f}'.format(accuracy_score(y_test, knn_pred)))

test_feature = test[feature_names]
y_pred = knn_model.predict(test_feature)
print(y_pred)
sample = pd.DataFrame()
sample['Booking_ID'] = test['Booking_ID']
sample['booking_status'] = y_pred
sample.to_csv('sample.csv', index=False)
```
