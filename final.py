
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
train['lead_nights'] = train['lead_time'] * train['nights']
train['lead_weekend_nights'] = train['lead_time'] * \
    train['no_of_weekend_nights']
train['lead_car'] = train['lead_time'] * \
    train['required_car_parking_space']
train['lead_room'] = train['lead_time'] * \
    train['room_type_reserved']
train['lead_market'] = train['lead_time'] * \
    train['repeated_guest']
train['lead_price'] = train['lead_time'] * \
    train['no_of_special_requests']
train['price_adults'] = train['avg_price_per_room'] * \
    train['no_of_adults']
train['price_room'] = train['avg_price_per_room'] * \
    train['room_type_reserved']
train['requests_room'] = train['no_of_special_requests'] * \
    train['room_type_reserved']
train['requests_lead'] = train['no_of_special_requests'] * \
    train['no_of_previous_cancellations']

train['no_of_adults'] += train['no_of_children']
train = train.drop(['no_of_children'], axis=1)

# test feature creation
test['nights'] = test['no_of_weekend_nights'] + test['no_of_week_nights']
test['lead_nights'] = test['lead_time'] * test['nights']
test['lead_weekend_nights'] = test['lead_time'] * test['no_of_weekend_nights']
test['lead_car'] = test['lead_time'] * \
    test['required_car_parking_space']
test['lead_room'] = test['lead_time'] * \
    test['room_type_reserved']
test['lead_market'] = test['lead_time'] * \
    test['repeated_guest']
test['lead_price'] = test['lead_time'] * \
    test['no_of_special_requests']
test['price_adults'] = test['avg_price_per_room'] * \
    test['no_of_adults']
test['price_room'] = test['avg_price_per_room'] * \
    test['room_type_reserved']
test['requests_room'] = test['no_of_special_requests'] * \
    test['room_type_reserved']
test['requests_lead'] = test['no_of_special_requests'] * \
    test['no_of_previous_cancellations']

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
    'lead_nights',
    'nights',
    'lead_weekend_nights',
    "lead_car",
    'lead_room',
    'lead_market',
    'lead_price',
    'price_adults',
    'price_room',
    'requests_room',
    'requests_lead'
]
scaler = MinMaxScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

train.to_csv('real_real_final_train.csv', index=False)
test.to_csv('real_real_final_test.csv', index=False)

# 모델링
feature_names = ['no_of_adults',
                 'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved',
                 'lead_time', 'market_segment_type',
                 'avg_price_per_room', 'no_of_special_requests',
                 'nights', 'lead_nights', 'lead_weekend_nights', "lead_car", 'lead_room', 'lead_market', 'lead_price', 'price_adults', 'price_room',
                 'requests_room', 'requests_lead']

X = train[feature_names]
y = train["booking_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# KNeighborsClassifier 학습/예측/평가
knn_model = KNeighborsClassifier(
    metric='manhattan', n_neighbors=28, weights='distance')

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
