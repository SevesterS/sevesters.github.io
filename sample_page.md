## Airbnb prices in Berlin

**Project description:** As we all have known, Airbnb has seen an ernomous growth with the number of rental listed on website growing exponentially each year. With the fast pace growth rate, it is not surprising for people to want to put up their free room/flat for rental on Airbnb website to earn an additional income. However, it is difficult for potential host to price their room because they might not know the true value of their home and how in-demand their home might be. Also, for existing host, some of them might have overpriced their room causing it to be unable to be rented out. Therefore, this project might address some of these problem. This project served to provide an in-depth insight to how valuable a room/flat might be depending of certain feature of their room/flat such as size, the distance of the room from central of the city and etc.


**Dataset:** In this project, I will be perfoming and in-depth analysis of Berlin. The dataset I will be using is from: https://www.kaggle.com/brittabettendorf/berlin-airbnb-data. Using only the listings_summary.csv


## Table of Content
**1. Viewing Data**
**2. Preprocessing of Data**
**3. Exploratary Data Analysis (EDA)
**4. Modelling the Data**
**5. Interprating Data**






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




### 2. Assess assumptions on which statistical inference will be based

```javascript
if (isAwesome){
  return true
}
```

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
