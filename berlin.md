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


### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
