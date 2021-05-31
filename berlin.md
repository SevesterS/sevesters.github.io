## Airbnb prices in Berlin

**Project description:** As we all have known, Airbnb has seen an ernomous growth with the number of rental listed on website growing exponentially each year. With the fast pace growth rate, it is not surprising for people to want to put up their free room/flat for rental on Airbnb website to earn an additional income. However, it is difficult for potential host to price their room because they might not know the true value of their home and how in-demand their home might be. Also, for existing host, some of them might have overpriced their room causing it to be unable to be rented out. Therefore, this project might address some of these problem. This project served to provide an in-depth insight to how valuable a room/flat might be depending of certain feature of their room/flat such as size, the distance of the room from central of the city and etc.


**Dataset:** In this project, I will be perfoming and in-depth analysis of Berlin. The dataset I will be using is from: https://www.kaggle.com/brittabettendorf/berlin-airbnb-data. Using only the listings_summary.csv


## Table of Content
<br>
1. Viewing Data
<br>
2. Preprocessing of Data
<br>
3. Exploratary Data Analysis (EDA)
<br>
4. Modelling the Data
<br>
5. Interprating Data
<br>





### 1. Viewing Data


```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
import lightgbm as lgb
import re
import xgboost as xgb


SEED = 42
```
```python
df_listing = pd.read_csv("C:/Users/Sevester Retseves/Desktop/SEVESTER/Data Analytics/Python/Capstone/Berlin/listings_summary.csv")
```
```pyton
#rows and columns
print("The dataset has {} rows and {} columns.".format(*df_listing.shape))
#duplicate
print("It contains {} duplicates.".format(df_listing.duplicated().sum()))
```
The dataset has 22552 rows and 96 columns.
It contains 0 duplicates.
```pyton
df_listing.info()
```
```pyton
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 22552 entries, 0 to 22551
Data columns (total 96 columns):
 #   Column                            Non-Null Count  Dtype  
---  ------                            --------------  -----  
 0   id                                22552 non-null  int64  
 1   listing_url                       22552 non-null  object 
 2   scrape_id                         22552 non-null  int64  
 3   last_scraped                      22552 non-null  object 
 4   name                              22493 non-null  object 
 5   summary                           21589 non-null  object 
 6   space                             14020 non-null  object 
 7   description                       22349 non-null  object 
 8   experiences_offered               22552 non-null  object 
 9   neighborhood_overview             11540 non-null  object 
 10  notes                             7215 non-null   object 
 11  transit                           13036 non-null  object 
 12  access                            10837 non-null  object 
 13  interaction                       10406 non-null  object 
 14  house_rules                       11449 non-null  object 
 15  thumbnail_url                     0 non-null      float64
 16  medium_url                        0 non-null      float64
 17  picture_url                       22552 non-null  object 
 18  xl_picture_url                    0 non-null      float64
 19  host_id                           22552 non-null  int64  
 20  host_url                          22552 non-null  object 
 21  host_name                         22526 non-null  object 
 22  host_since                        22526 non-null  object 
 23  host_location                     22436 non-null  object 
 24  host_about                        11189 non-null  object 
 25  host_response_time                9658 non-null   object 
 26  host_response_rate                9657 non-null   object 
 27  host_acceptance_rate              0 non-null      float64
 28  host_is_superhost                 22526 non-null  object 
 29  host_thumbnail_url                22526 non-null  object 
 30  host_picture_url                  22526 non-null  object 
 31  host_neighbourhood                17458 non-null  object 
 32  host_listings_count               22526 non-null  float64
 33  host_total_listings_count         22526 non-null  float64
 34  host_verifications                22552 non-null  object 
 35  host_has_profile_pic              22526 non-null  object 
 36  host_identity_verified            22526 non-null  object 
 37  street                            22552 non-null  object 
 38  neighbourhood                     21421 non-null  object 
 39  neighbourhood_cleansed            22552 non-null  object 
 40  neighbourhood_group_cleansed      22552 non-null  object 
 41  city                              22547 non-null  object 
 42  state                             22468 non-null  object 
 43  zipcode                           21896 non-null  object 
 44  market                            22489 non-null  object 
 45  smart_location                    22552 non-null  object 
 46  country_code                      22552 non-null  object 
 47  country                           22552 non-null  object 
 48  latitude                          22552 non-null  float64
 49  longitude                         22552 non-null  float64
 50  is_location_exact                 22552 non-null  object 
 51  property_type                     22552 non-null  object 
 52  room_type                         22552 non-null  object 
 53  accommodates                      22552 non-null  int64  
 54  bathrooms                         22520 non-null  float64
 55  bedrooms                          22534 non-null  float64
 56  beds                              22512 non-null  float64
 57  bed_type                          22552 non-null  object 
 58  amenities                         22552 non-null  object 
 59  square_feet                       446 non-null    float64
 60  price                             22552 non-null  object 
 61  weekly_price                      3681 non-null   object 
 62  monthly_price                     2659 non-null   object 
 63  security_deposit                  13191 non-null  object 
 64  cleaning_fee                      15406 non-null  object 
 65  guests_included                   22552 non-null  int64  
 66  extra_people                      22552 non-null  object 
 67  minimum_nights                    22552 non-null  int64  
 68  maximum_nights                    22552 non-null  int64  
 69  calendar_updated                  22552 non-null  object 
 70  has_availability                  22552 non-null  object 
 71  availability_30                   22552 non-null  int64  
 72  availability_60                   22552 non-null  int64  
 73  availability_90                   22552 non-null  int64  
 74  availability_365                  22552 non-null  int64  
 75  calendar_last_scraped             22552 non-null  object 
 76  number_of_reviews                 22552 non-null  int64  
 77  first_review                      18638 non-null  object 
 78  last_review                       18644 non-null  object 
 79  review_scores_rating              18163 non-null  float64
 80  review_scores_accuracy            18138 non-null  float64
 81  review_scores_cleanliness         18141 non-null  float64
 82  review_scores_checkin             18120 non-null  float64
 83  review_scores_communication       18134 non-null  float64
 84  review_scores_location            18121 non-null  float64
 85  review_scores_value               18117 non-null  float64
 86  requires_license                  22552 non-null  object 
 87  license                           1638 non-null   object 
 88  jurisdiction_names                0 non-null      float64
 89  instant_bookable                  22552 non-null  object 
 90  is_business_travel_ready          22552 non-null  object 
 91  cancellation_policy               22552 non-null  object 
 92  require_guest_profile_picture     22552 non-null  object 
 93  require_guest_phone_verification  22552 non-null  object 
 94  calculated_host_listings_count    22552 non-null  int64  
 95  reviews_per_month                 18638 non-null  float64
dtypes: float64(21), int64(13), object(62)
memory usage: 16.5+ MB
```
```python
df_listing.head(1)
```
<img src="images/heading.PNG?raw=true"/>

```python
df_listing.columns
```

<img src="images/column.PNG?raw=true"/>



### 2. Preprocessing Data

```python
#Choosing which columns to keep
columns_to_keep = ['id', 'space', 'description', 'host_has_profile_pic', 'neighbourhood_group_cleansed', 
                   'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',  
                   'bedrooms', 'bed_type', 'amenities', 'square_feet', 'price', 'cleaning_fee', 
                   'security_deposit', 'extra_people', 'guests_included', 'minimum_nights',  
                   'instant_bookable', 'is_business_travel_ready', 'cancellation_policy']

df_raw = df_listing[columns_to_keep].set_index('id')
print("The dataset has {} rows and {} columns - after dropping irrelevant columns.".format(*df_raw.shape))
```
The dataset has 22552 rows and 23 columns - after dropping irrelevant columns.
```python
#how many room type
df_raw.room_type.value_counts(normalize=True)
```
Private room       0.511440
<br>
Entire home/apt    0.475435
<br>
Shared room        0.013125
<br>
Name: room_type, dtype: float64
```python
#missing value
df_raw.isna().sum()
```
```
space                            8532
description                       203
host_has_profile_pic               26
neighbourhood_group_cleansed        0
latitude                            0
longitude                           0
property_type                       0
room_type                           0
accommodates                        0
bathrooms                          32
bedrooms                           18
bed_type                            0
amenities                           0
square_feet                     22106
price                               0
cleaning_fee                     7146
security_deposit                 9361
extra_people                        0
guests_included                     0
minimum_nights                      0
instant_bookable                    0
is_business_travel_ready            0
cancellation_policy                 0
dtype: int64
```
```python
#replace cleaning_fee's null value to $0.00
df_raw.cleaning_fee.fillna('$0.00', inplace=True)
df_raw.cleaning_fee.isna().sum()
```
0
```python
#replcae security_deposit's null value to $0.00
df_raw.security_deposit.fillna('$0.00', inplace=True)
df_raw.security_deposit.isna().sum()
```
0
```python
#remove $ and change to float type
df_raw.price = df_raw.price.str.replace('$', '').str.replace(',', '').astype(float)
df_raw.cleaning_fee = df_raw.cleaning_fee.str.replace('$', '').str.replace(',', '').astype(float)
df_raw.security_deposit = df_raw.security_deposit.str.replace('$', '').str.replace(',', '').astype(float)
df_raw.extra_people = df_raw.extra_people.str.replace('$', '').str.replace(',', '').astype(float)
```
```python
#delete price > $500 and price = $0
df_raw.drop(df_raw[ (df_raw.price > 500) | (df_raw.price == 0) ].index, axis=0, inplace=True)
```
```python
#delete columns with too many missing values
df_raw.drop(columns = ['space', 'square_feet'], inplace= True)
```
```python
#data after processed
print("The dataset has {} rows and {} columns".format(*df_raw.shape))
```
The dataset has 22420 rows and 21 columns

### 2.1 Feature Engineering (Distance to central of Berlin)
```python
from geopy.distance import great_circle
```
```python
def distance_to_mid(lat, lon):
    berlin_centre = (52.5027778, 13.404166666666667)
    airbnb = (lat, lon)
    return great_circle(berlin_centre, airbnb).km
```
```python
df_raw['distance_in_km'] = df_raw.apply(lambda x: distance_to_mid(x.latitude, x.longitude), axis=1)
```
```python
df_raw.head(3)
```

<img src="images/distance.PNG?raw=true"/>

### 2.2 Feature Engineering (Lodging Size)
```python
#extract numbers from description
df_raw['size'] = df_raw['description'].str.extract('(\d{2,3}\s?[smSM])', expand=True)
df_raw['size'] = df_raw['size'].str.replace("\D", "")

#change datatype of size into float
df_raw['size'] = df_raw['size'].astype(float)
```
```python
df_raw[['description', 'size']].head(3)
```
<img src="images/size.PNG?raw=true"/>

```python
df_raw.isna().sum()
```


```
description                       202
host_has_profile_pic                0
neighbourhood_group_cleansed        0
latitude                            0
longitude                           0
property_type                       0
room_type                           0
accommodates                        0
bathrooms                           0
bedrooms                            0
bed_type                            0
amenities                           0
price                               0
cleaning_fee                        0
security_deposit                    0
extra_people                        0
guests_included                     0
minimum_nights                      0
instant_bookable                    0
is_business_travel_ready            0
cancellation_policy                 0
distance_in_km                      0
size                            11735
dtype: int64
```
```python
predict_df = df_raw[['accommodates', 'bathrooms', 'bedrooms', 'price', 'cleaning_fee', 'security_deposit', 'extra_people', 'guests_included', 'distance_in_km', 'size']] 
```
```python
# split datasets
train_data = predict_df[predict_df['size'].notnull()]
test_data  = predict_df[predict_df['size'].isnull()]

# define X
X_train = train_data.drop('size', axis=1)
X_test  = test_data.drop('size', axis=1)

# define y
y_train = train_data['size']
```
```python
# import Linear Regression
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

# fit model to training data
linreg.fit(X_train, y_train)
```
```python
# making predictions
y_test = linreg.predict(X_test)
```
```python
y_test = pd.DataFrame(y_test)
y_test.columns = ['size']
print(y_test.shape)
y_test.head()
```

<img src="images/pred.PNG?raw=true"/>

```python
# make the index of X_test to an own dataframe
prelim_index = pd.DataFrame(X_test.index)
prelim_index.columns = ['prelim']

# ... and concat this dataframe with y_test
y_test = pd.concat([y_test, prelim_index], axis=1)
y_test.set_index(['prelim'], inplace=True)
y_test.head()
```


<img src="images/prelim.PNG?raw=true"/>



```python
new_test_data = pd.concat([X_test, y_test], axis=1)
```
```python
print(new_test_data.shape)
new_test_data.head()
```
(11735, 10)
<img src="images/concat.PNG?raw=true"/>

```python
new_test_data['size'].isna().sum()
```
0

```python
# combine train and test data back to a new predict df
predict_df_new = pd.concat([new_test_data, train_data], axis=0)

print(predict_df_new.shape)
predict_df_new.head()
```
(22420, 10)

```python
# prepare the multiple columns before concatening
df_raw.drop(['accommodates', 'bathrooms', 'bedrooms', 'price', 'cleaning_fee', 
             'security_deposit', 'extra_people', 'guests_included', 'distance_in_km', 'size'], 
            axis=1, inplace=True)
```

```python
# concate back to complete dataframe
df = pd.concat([predict_df_new, df_raw], axis=1)

print(df.shape)
df.head(2)
```
```python
#delete size > $500 and size = $0
df.drop(df[ (df['size'] > 500) | (df['size'] == 0) ].index, axis=0, inplace=True)
```






### 3. EDA

```python
#Price difference on MAP

df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(10,7), 
        c="price", cmap="gist_heat_r", colorbar=True, sharex=False);
```

<img src="images/heatmap.PNG?raw=true"/>

```python
relation = df.plot.scatter('distance_in_km', 'price', title = 'Relation Between Price and distance')
```

<img src="images/relation.PNG?raw=true"/>

```python
sns.set_style("white")
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

fig, ax = plt.subplots(figsize=(12,7))
ax = sns.scatterplot(x="size", y="price", size='cleaning_fee', sizes=(5, 200),
                      hue='size', palette=cmap,  data=df)

plt.title('\nRelation between Size & Price\n', fontsize=14, fontweight='bold')

# putting legend out of the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
```

<img src="images/sp.PNG?raw=true"/>

```python
plt.figure(figsize=(6,6))
sns.heatmap(df.groupby(['neighbourhood_group_cleansed', 'bedrooms']).price.median().unstack(), 
            cmap='Reds', annot=True, fmt=".0f")

plt.xlabel('\nBedrooms', fontsize=12)
plt.ylabel('District\n', fontsize=12)
plt.title('\nHeatmap: Prices by Neighbourhood and Number of Bedrooms\n\n', fontsize=14, fontweight='bold');
```

<img src="images/pb.PNG?raw=true"/>


### 4.1 Modelling Data (Linear Regression)

```python
# 1)

X = df.drop(['price',
             'host_has_profile_pic',
             'room_type',
             'bed_type',
             'instant_bookable',
             'is_business_travel_ready',
             'cancellation_policy',
             'TV',
             'Wifi',
             'Smoking_allowed'], axis=1) # input
y = df['price'] # output (dependent variable)
```
```python
# 2) Splitting our data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False, random_state=SEED)
```
```python
# scale data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
```
```python
# Import the linear regression algorithm
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# Train the model
regressor.fit(X_train, y_train)
```
```python
# Kept aside some data to test - X_test
y_pred = regressor.predict(X_test)

compare_df = pd.DataFrame({"Desired Output (Actuals)": y_test, 
                           "Predicted Output": y_pred})
```
```python
compare_df
```

<img src="images/compare.PNG?raw=true"/>

```python
# The coefficients
print('Coefficients: \n', regressor.coef_)

# The mean squared error
print('Mean squared error: {:.2f}'.format(mean_squared_error(y_test, y_pred)))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: {:.2f}'.format(r2_score(y_test, y_pred)))
```
```Coefficients: 
 [13.53791532  3.07493434  6.32252553  7.79885889  1.83117465 -0.42416653
  5.07124663 -3.62918189  5.48296261  0.04768222]
Mean squared error: 1698.28
Coefficient of determination: 0.29
```
```python
# Evaluate the model's training score and test score
print("Regression model's training score = {:.2f}".format(regressor.score(X_train, y_train)))
print("Regression model's test score     = {:.2f}".format(regressor.score(X_test, y_test)))
```
Regression model's training score = 0.46
<br>
Regression model's test score     = 0.29

```python
own_pred = regressor.predict(X_test)
print("My target value is   =", str(own_pred[0]))
print("My observed value is =", str(y_test.iloc[0]))
```
My target value is   = 25.288911152484022\
<br>
My observed value is = 20.0

```python
# get importance
importance = regressor.coef_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
```
```
Feature: 0, Score: 13.53792
Feature: 1, Score: 3.07493
Feature: 2, Score: 6.32253
Feature: 3, Score: 7.79886
Feature: 4, Score: 1.83117
Feature: 5, Score: -0.42417
Feature: 6, Score: 5.07125
Feature: 7, Score: -3.62918
Feature: 8, Score: 5.48296
Feature: 9, Score: 0.04768
```
```python
# plot feature importance
#plt.bar([x for x in range(len(importance))], importance)
plt.figure(figsize=(10,5))
plt.bar([x for x in X], importance)
plt.xticks(rotation=90)
plt.show()
```

<img src="images/reg.PNG?raw=true"/>




### 4.2 Modelling Data (Random Forest)

```python
X = features_coded # input
y = target # output (dependent variable)
```
```python
# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
y = np.array(df['price'])
# Remove the labels from the features
# axis 1 refers to the columns
X = df.drop(['price',
             'host_has_profile_pic',
             'room_type',
             'bed_type',
             'instant_bookable',
             'is_business_travel_ready',
             'cancellation_policy',
             'TV',
             'Wifi',
             'Smoking_allowed'], axis = 1)
# Saving feature names for later use
X_list = list(X.columns)
# Convert to numpy array
X = np.array(X)
```
```python
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = SEED)
```
```python
#checking the shape
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', X_test.shape)
print('Testing Features Shape:', y_train.shape)
print('Testing Labels Shape:', y_test.shape)
```
```
Training Features Shape: (17711, 10)
Training Labels Shape: (4428, 10)
Testing Features Shape: (17711,)
Testing Labels Shape: (4428,)
```

```python
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = SEED)
# Train the model on training data
rf.fit(X, y);
```

```python
rf.get_params()
```
```
{'bootstrap': True,
 'ccp_alpha': 0.0,
 'criterion': 'mse',
 'max_depth': None,
 'max_features': 'auto',
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 1000,
 'n_jobs': None,
 'oob_score': False,
 'random_state': 42,
 'verbose': 0,
 'warm_start': False}
```
```python
y_pred = rf.predict(X_test)

compare_df = pd.DataFrame({"Desired Output (Actuals)": y_test, 
                           "Predicted Output": y_pred})
```
```python
compare_df[:10]
```
<img src="images/comp.PNG?raw=true"/>

```python
# The mean squared error
print('Mean squared error: {:.2f}'.format(mean_squared_error(y_test, y_pred)))
```
Mean squared error: 106.04

```python
# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))
```
Mean Absolute Error: 4.92

```python
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
```
Accuracy: 90.94 %.

```python
# get importance
importance = rf.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
```
```
Feature: 0, Score: 0.11010
Feature: 1, Score: 0.03137
Feature: 2, Score: 0.07317
Feature: 3, Score: 0.07050
Feature: 4, Score: 0.02756
Feature: 5, Score: 0.05583
Feature: 6, Score: 0.02783
Feature: 7, Score: 0.11243
Feature: 8, Score: 0.44544
Feature: 9, Score: 0.04577
```
```python
plt.barh(X_list, rf.feature_importances_)
```

<img src="images/feature.PNG?raw=true"/>

## Hyperparameter tuning RandomSearch with Cross Validation

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = SEED)
from pprint import pprint
# Look at parameters used by our current forest
rf = rf.fit(X_train, y_train)
print('Parameters currently in use:\n')
pprint(rf.get_params())
```
```
Parameters currently in use:

{'bootstrap': True,
 'ccp_alpha': 0.0,
 'criterion': 'mse',
 'max_depth': None,
 'max_features': 'auto',
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 100,
 'n_jobs': None,
 'oob_score': False,
 'random_state': 42,
 'verbose': 0,
 'warm_start': False}
```

```python
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
```
```
{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
```
```python
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=SEED, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
```

<img src="images/random.PNG?raw=true"/>


```python
rf_random.best_score_
```
0.6241257650995122

```python
# Use the forest's predict method on the test data
predictions = rf_random.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))
```
Mean Absolute Error: 13.92
```python
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
```
Accuracy: 73.81 %.


## Hyperparameter tuning GridSearch with Cross Validation

```python
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [None],
    'max_features': ['auto'],
    'min_samples_leaf': [0, 1, 2],
    'min_samples_split': [0, 1, 2],
    'n_estimators': [100, 200, 300, 400]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
```

```python
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_
```

```python
grid_search.best_score_
```
0.6227696457949264

```python
# Use the forest's predict method on the test data
predictions = grid_search.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))
```
Mean Absolute Error: 13.5

```python
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
```
Accuracy: 75.0 %.

### 5. Interprating Data


It turns out that the price is dependent not only on geography. In fact, the most important feature affecting the price of the room/flat is the size.
More than 50% of the price is attributed by the size and distance from central of Berlin, 44% and 11% respectively.

Therefore with Random Forest is a better modelling for price prediction in this case as it can accurately predict correctly 90% of the time.
With this in mind, there will no longer be as much over-priced room so people can afford it better and rooms will be able to be rented out easier.
Potential new hosts are able to put up their rooms/flats up on Airbnb website with confidence with the knowledge of how much their rooms truly worth.

The next step would be good to start all over again and include the features like the reviews from consumer to try to find out if accuracy improves. That might help a beginner on Airbnb better know what price to aim for.



