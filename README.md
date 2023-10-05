# AirBnB dataset 

**Course:** MGOC15 - Introductory Business Data Analytics <br>
**Topics Tested:** Modifying Dataframes, Categorical Imputation, Outliers <br>


### Florence AirBnB Bookings

We have a dataset called "AirBnB.csv". Some important columns: <br>

**BookingsPerMonth** - denotes the average number of bookings a property has received in a given month (Since this denotes the total number of bookings divided by the time period, it is likely to be a fraction).

**CommissionsPerMonth** - how much airbnb collects in commissions from a given property.

### Loading Python modules

We start by first importing packages and modules needed for our analysis


```python
import pandas as pd               # for data manipulation
import numpy as np                # for stats and numerical analysis
import matplotlib.pyplot as plt   # for plotting and data visualization

# Disable SetttingwithCopyWarning
# Don't worry about why this line is included
pd.set_option('mode.chained_assignment', None)
```

### Read the file before proceeding


```python
df_airbnb = pd.read_csv('AirBnB.csv')
df_airbnb
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HostResponseTime</th>
      <th>HostResponseRate</th>
      <th>HostAcceptRate</th>
      <th>Superhost</th>
      <th>HostListings</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>RoomType</th>
      <th>Accomodates</th>
      <th>...</th>
      <th>AvgRating</th>
      <th>RatingAccuracy</th>
      <th>RatingClean</th>
      <th>RatingCheckIn</th>
      <th>RatingCommunication</th>
      <th>RatingLocation</th>
      <th>RatingValue</th>
      <th>Instant Booking</th>
      <th>BookingsPerMonth</th>
      <th>CommissionsPerMonth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>within an hour</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>f</td>
      <td>32</td>
      <td>Centro Storico</td>
      <td>43.777090</td>
      <td>11.252160</td>
      <td>Private room</td>
      <td>2</td>
      <td>...</td>
      <td>4.65</td>
      <td>4.73</td>
      <td>4.87</td>
      <td>4.85</td>
      <td>4.91</td>
      <td>4.90</td>
      <td>4.71</td>
      <td>t</td>
      <td>1.19</td>
      <td>34.8075</td>
    </tr>
    <tr>
      <th>1</th>
      <td>within an hour</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>f</td>
      <td>5</td>
      <td>Centro Storico</td>
      <td>43.762680</td>
      <td>11.241490</td>
      <td>Hotel room</td>
      <td>2</td>
      <td>...</td>
      <td>4.84</td>
      <td>4.82</td>
      <td>4.87</td>
      <td>4.88</td>
      <td>4.92</td>
      <td>4.71</td>
      <td>4.82</td>
      <td>t</td>
      <td>4.58</td>
      <td>144.2700</td>
    </tr>
    <tr>
      <th>2</th>
      <td>within an hour</td>
      <td>1.00</td>
      <td>0.83</td>
      <td>f</td>
      <td>10</td>
      <td>Centro Storico</td>
      <td>43.775250</td>
      <td>11.252580</td>
      <td>Private room</td>
      <td>2</td>
      <td>...</td>
      <td>4.47</td>
      <td>4.60</td>
      <td>4.65</td>
      <td>4.79</td>
      <td>4.72</td>
      <td>4.88</td>
      <td>4.49</td>
      <td>f</td>
      <td>0.65</td>
      <td>14.6250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>within an hour</td>
      <td>1.00</td>
      <td>0.83</td>
      <td>f</td>
      <td>10</td>
      <td>Centro Storico</td>
      <td>43.775250</td>
      <td>11.252580</td>
      <td>Private room</td>
      <td>2</td>
      <td>...</td>
      <td>4.08</td>
      <td>3.83</td>
      <td>4.33</td>
      <td>4.58</td>
      <td>4.42</td>
      <td>5.00</td>
      <td>4.17</td>
      <td>f</td>
      <td>0.56</td>
      <td>15.1200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>within an hour</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>f</td>
      <td>189</td>
      <td>Centro Storico</td>
      <td>43.769440</td>
      <td>11.263310</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>...</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.50</td>
      <td>3.50</td>
      <td>4.50</td>
      <td>4.00</td>
      <td>t</td>
      <td>0.03</td>
      <td>1.4445</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5669</th>
      <td>within an hour</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>f</td>
      <td>14</td>
      <td>Rifredi</td>
      <td>43.794050</td>
      <td>11.240850</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>...</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>t</td>
      <td>1.00</td>
      <td>23.8500</td>
    </tr>
    <tr>
      <th>5670</th>
      <td>within an hour</td>
      <td>0.97</td>
      <td>1.00</td>
      <td>t</td>
      <td>8</td>
      <td>Isolotto Legnaia</td>
      <td>43.766420</td>
      <td>11.238470</td>
      <td>Private room</td>
      <td>2</td>
      <td>...</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>t</td>
      <td>1.00</td>
      <td>17.5500</td>
    </tr>
    <tr>
      <th>5671</th>
      <td>within a day</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>f</td>
      <td>0</td>
      <td>Campo di Marte</td>
      <td>43.785920</td>
      <td>11.289180</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>...</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>4.67</td>
      <td>4.67</td>
      <td>t</td>
      <td>3.00</td>
      <td>135.0000</td>
    </tr>
    <tr>
      <th>5672</th>
      <td>within a few hours</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>f</td>
      <td>0</td>
      <td>Campo di Marte</td>
      <td>43.787490</td>
      <td>11.264180</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>...</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>t</td>
      <td>1.00</td>
      <td>14.4000</td>
    </tr>
    <tr>
      <th>5673</th>
      <td>within an hour</td>
      <td>1.00</td>
      <td>0.77</td>
      <td>f</td>
      <td>0</td>
      <td>Rifredi</td>
      <td>43.790207</td>
      <td>11.251734</td>
      <td>Private room</td>
      <td>2</td>
      <td>...</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>f</td>
      <td>1.00</td>
      <td>12.1500</td>
    </tr>
  </tbody>
</table>
<p>5674 rows × 31 columns</p>
</div>



### Modifying data

a) The residents in the historic Centro Storico neighbourhood are complaining that AirBnB is extracting too much of their revenue as commission. In response, AirBnB has instituted a new rule where they will no longer collect more than \$1000 a month from any property in the Centro Storico neighborhood., i.e., all commissions are capped at \\$1000 for this neighbourhood alone. The excess money has been refunded to all hosts. Please make the changes in the *CommissionsPerMonth* column to reflect this.

b) some houses have listed the number of bathrooms as 1.5 and these need to be rounded up to 2. **Use df.loc to solve this problem**.


```python
#Solution for Part a
df_airbnb.loc[(df_airbnb['CommissionsPerMonth']>1000), 'CommissionsPerMonth']=1000
```


```python
#Solution for Part b
df_airbnb.loc[(df_airbnb['Bathrooms'] == 1.5), 'Bathrooms']=2
```

### Outliers

One of our future goals is to understand how the booking rate depends on features such as Average Rating (AvgRating). Before we start studying correlations, however, we need to check whether these two fields (BookingsPerMonth/AvgRating) have any outliers. If so, deal with them before proceeding.

Use both visualizations and statistics to answer the question. This question only requires you to remove the corresponding outliers.

**Follow-up Question:** Can categorical columns have outliers? Explain why/why not.


```python
#Answer 2a
# First, let's visualize the 'AvgRating' column
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
df_airbnb['AvgRating'].plot.box()
plt.subplot(1, 2, 2)
df_airbnb['AvgRating'].plot.hist()
```




    <AxesSubplot: ylabel='Frequency'>




    
![png](/images/output_9_1.png)
    



```python
#The 'AvgRating' graphs show that AvgRating is left-skewed. Let's use statistics to identify outliers. 
```


```python
ratingThr = df_airbnb['AvgRating'].mean() - 5*df_airbnb['AvgRating'].std()
```


```python
# let's visualize the 'BookingsPerMonth' column
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
df_airbnb['BookingsPerMonth'].plot.box()
plt.subplot(1, 2, 2)
df_airbnb['BookingsPerMonth'].plot.hist()
```




    <AxesSubplot: ylabel='Frequency'>




    
![png](/images/output_12_1.png)
    



```python
#The 'BookingsPerMonth' graphs show that BookingsPerMonth is right-skewed. Let's use statistics to identify outliers. 
```


```python
bookingThr = df_airbnb['BookingsPerMonth'].mean() + 5*df_airbnb['BookingsPerMonth'].std()
```


```python
df_noRat = df_airbnb[ df_airbnb['AvgRating'] > ratingThr]
df_noOut = df_noRat[df_noRat['BookingsPerMonth'] < bookingThr]
```

#Answer 2b (Follow-up Question)
The question of whether a categorical data contains an outlier depends on how you approach it and what you want to do with the data. It can be said that, for a column to have an outlier, there should be a quantitative measure that can be used to compare/contrast. Since categorical columns do not contain quantitative data, they cannot have outliers. However, if a category was miscategorized (e.g., a color that doesn't exist) and, therefore, have no or very few values in the data, then it could be taken as an outlier. 

### Categorical Imputation

Unfortunately, the dataset is missing several entries for its Price column (we do not wish to delete them due to potential bias). Instead of simply replacing everything with the average price, you proposed a more nuanced approach where you would use the number of Bedrooms in the listing to determine its price --- after all, won't most 2 bedroom houses be priced similarly? 

Your co-worker on the other hand is skeptical of this approach because houses with the same number of bedrooms can be priced very differently based on other features. 

a) First, convince your co-worker that houses with more bedrooms have a higher average price, i.e., increasing the number of bedrooms leads to a higher average price.

b) Assuming that you were successful in convincing her, use categorical imputation based on the number of bedrooms to fill in the missing price entries.

(There is no need to remove any additional outliers for this question)

    Hint: If you are lazy, you can try a for loop.


```python
#Answer for Part a
#We will divide the data into two categories: houses with more bedrooms vs. houses with fewer bedrooms. 
#but for doing do, we need to identify a split we can use for dividing it into the two categories 
#let's first get generic information using the function 'describe'
df_noOut['Bedrooms'].describe()
```




    count    5399.000000
    mean        1.566401
    std         0.865110
    min         1.000000
    25%         1.000000
    50%         1.000000
    75%         2.000000
    max         9.000000
    Name: Bedrooms, dtype: float64




```python
# 2 should be a good number for the split 
split = 2
dfMore = df_noOut[ df_noOut['Bedrooms'] > split ]
dfFew = df_noOut[df_noOut['Bedrooms'] <= split]

MorePrice = dfMore['Price'].mean()
FewPrice = dfFew['Price'].mean()

print('Average price of houses with more than 2 bedrooms is ', MorePrice)
print('Average price of houses with 2 or fewer bedrooms is ', FewPrice)
print('The statement is indeed true, i.e., the average price of houses with more bedrooms is indeed higher than those with fewer bedrooms')
```

    Average price of houses with more than 2 bedrooms is  197.79443585780527
    Average price of houses with 2 or fewer bedrooms is  85.99259415996615
    The statement is indeed true, i.e., the average price of houses with more bedrooms is indeed higher than those with fewer bedrooms



```python
#Answer for Part b
df_noOut.loc[(df_noOut['Price'].isnull()) & (df_noOut['Bedrooms'] > split), 'Price'] = MorePrice
df_noOut.loc[(df_noOut['Price'].isnull()) & (df_noOut['Bedrooms'] <= split), 'Price'] = FewPrice
```

### Numeric to Categorical Columns 

The city wants to divide the AirBnBs into high-volume homes (lots of bookings) and low-volume homes (fewer bookings). Categorical columns are typically much easier to deal with than numeric columns. With this in mind, create a new column called **cat_booking** - the column must equal one (or True) if the property is a high-volume property and equal 0 (or False) if it is a low-volume property. 

Use any suitable threshold for the *BookingsPerMonth* column as discussed in class.


```python
#Answer for Q4
df_noOut['BookingsPerMonth'].describe()
```




    count    5628.000000
    mean        1.449733
    std         1.435517
    min         0.010000
    25%         0.300000
    50%         1.000000
    75%         2.160000
    max         8.750000
    Name: BookingsPerMonth, dtype: float64




```python
BookingSplit = 2
df_noOut['cat_booking'] = df_noOut['BookingsPerMonth'] > BookingSplit
```


```python

```
