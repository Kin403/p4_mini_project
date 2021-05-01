## Problem 4

##### In this section, I will be using the COVID dataset (owid-covid-data.csv) used in problem 3.1 to write my data-driven blog post.

## After the chaos - A glance to the impacts of COVID-19

##### 2020 is a year we will never forget. The COVID-19 pandemic has stroke us seriously from the beginning of the year, and the world has come to a sudden pause.

##### A year later, new vaccines are invented, yet, we are still adopting a new living and working style.

##### The pandemic has taught us some lessons in a tough way, and more than 140 millions people are infected up till now. 

##### In this blog post, I will try to briefly investigate the impacts of the pandemic by showing different figures and graphs. 


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df=pd.read_csv('owid-covid-data.csv')
```


```python
world_data=df[(df['location'].str.contains('World'))]
world_data
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
      <th>iso_code</th>
      <th>continent</th>
      <th>location</th>
      <th>date</th>
      <th>total_cases</th>
      <th>new_cases</th>
      <th>new_cases_smoothed</th>
      <th>total_deaths</th>
      <th>new_deaths</th>
      <th>new_deaths_smoothed</th>
      <th>...</th>
      <th>gdp_per_capita</th>
      <th>extreme_poverty</th>
      <th>cardiovasc_death_rate</th>
      <th>diabetes_prevalence</th>
      <th>female_smokers</th>
      <th>male_smokers</th>
      <th>handwashing_facilities</th>
      <th>hospital_beds_per_thousand</th>
      <th>life_expectancy</th>
      <th>human_development_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>75317</td>
      <td>OWID_WRL</td>
      <td>NaN</td>
      <td>World</td>
      <td>2020-01-22</td>
      <td>557.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>15469.207</td>
      <td>10.0</td>
      <td>233.07</td>
      <td>8.51</td>
      <td>6.434</td>
      <td>34.635</td>
      <td>60.13</td>
      <td>2.705</td>
      <td>72.58</td>
      <td>0.737</td>
    </tr>
    <tr>
      <td>75318</td>
      <td>OWID_WRL</td>
      <td>NaN</td>
      <td>World</td>
      <td>2020-01-23</td>
      <td>655.0</td>
      <td>98.0</td>
      <td>NaN</td>
      <td>18.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>15469.207</td>
      <td>10.0</td>
      <td>233.07</td>
      <td>8.51</td>
      <td>6.434</td>
      <td>34.635</td>
      <td>60.13</td>
      <td>2.705</td>
      <td>72.58</td>
      <td>0.737</td>
    </tr>
    <tr>
      <td>75319</td>
      <td>OWID_WRL</td>
      <td>NaN</td>
      <td>World</td>
      <td>2020-01-24</td>
      <td>941.0</td>
      <td>286.0</td>
      <td>NaN</td>
      <td>26.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>15469.207</td>
      <td>10.0</td>
      <td>233.07</td>
      <td>8.51</td>
      <td>6.434</td>
      <td>34.635</td>
      <td>60.13</td>
      <td>2.705</td>
      <td>72.58</td>
      <td>0.737</td>
    </tr>
    <tr>
      <td>75320</td>
      <td>OWID_WRL</td>
      <td>NaN</td>
      <td>World</td>
      <td>2020-01-25</td>
      <td>1433.0</td>
      <td>492.0</td>
      <td>NaN</td>
      <td>42.0</td>
      <td>16.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>15469.207</td>
      <td>10.0</td>
      <td>233.07</td>
      <td>8.51</td>
      <td>6.434</td>
      <td>34.635</td>
      <td>60.13</td>
      <td>2.705</td>
      <td>72.58</td>
      <td>0.737</td>
    </tr>
    <tr>
      <td>75321</td>
      <td>OWID_WRL</td>
      <td>NaN</td>
      <td>World</td>
      <td>2020-01-26</td>
      <td>2118.0</td>
      <td>685.0</td>
      <td>NaN</td>
      <td>56.0</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>15469.207</td>
      <td>10.0</td>
      <td>233.07</td>
      <td>8.51</td>
      <td>6.434</td>
      <td>34.635</td>
      <td>60.13</td>
      <td>2.705</td>
      <td>72.58</td>
      <td>0.737</td>
    </tr>
    <tr>
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
      <td>...</td>
    </tr>
    <tr>
      <td>75739</td>
      <td>OWID_WRL</td>
      <td>NaN</td>
      <td>World</td>
      <td>2021-03-19</td>
      <td>122315656.0</td>
      <td>526273.0</td>
      <td>465024.714</td>
      <td>2701446.0</td>
      <td>10062.0</td>
      <td>8791.143</td>
      <td>...</td>
      <td>15469.207</td>
      <td>10.0</td>
      <td>233.07</td>
      <td>8.51</td>
      <td>6.434</td>
      <td>34.635</td>
      <td>60.13</td>
      <td>2.705</td>
      <td>72.58</td>
      <td>0.737</td>
    </tr>
    <tr>
      <td>75740</td>
      <td>OWID_WRL</td>
      <td>NaN</td>
      <td>World</td>
      <td>2021-03-20</td>
      <td>122813915.0</td>
      <td>498259.0</td>
      <td>471290.429</td>
      <td>2709640.0</td>
      <td>8194.0</td>
      <td>8716.286</td>
      <td>...</td>
      <td>15469.207</td>
      <td>10.0</td>
      <td>233.07</td>
      <td>8.51</td>
      <td>6.434</td>
      <td>34.635</td>
      <td>60.13</td>
      <td>2.705</td>
      <td>72.58</td>
      <td>0.737</td>
    </tr>
    <tr>
      <td>75741</td>
      <td>OWID_WRL</td>
      <td>NaN</td>
      <td>World</td>
      <td>2021-03-21</td>
      <td>123207780.0</td>
      <td>393865.0</td>
      <td>476093.571</td>
      <td>2715295.0</td>
      <td>5655.0</td>
      <td>8739.286</td>
      <td>...</td>
      <td>15469.207</td>
      <td>10.0</td>
      <td>233.07</td>
      <td>8.51</td>
      <td>6.434</td>
      <td>34.635</td>
      <td>60.13</td>
      <td>2.705</td>
      <td>72.58</td>
      <td>0.737</td>
    </tr>
    <tr>
      <td>75742</td>
      <td>OWID_WRL</td>
      <td>NaN</td>
      <td>World</td>
      <td>2021-03-22</td>
      <td>123678061.0</td>
      <td>470281.0</td>
      <td>493415.571</td>
      <td>2722975.0</td>
      <td>7680.0</td>
      <td>8864.143</td>
      <td>...</td>
      <td>15469.207</td>
      <td>10.0</td>
      <td>233.07</td>
      <td>8.51</td>
      <td>6.434</td>
      <td>34.635</td>
      <td>60.13</td>
      <td>2.705</td>
      <td>72.58</td>
      <td>0.737</td>
    </tr>
    <tr>
      <td>75743</td>
      <td>OWID_WRL</td>
      <td>NaN</td>
      <td>World</td>
      <td>2021-03-23</td>
      <td>124202136.0</td>
      <td>524075.0</td>
      <td>500731.000</td>
      <td>2734098.0</td>
      <td>11123.0</td>
      <td>9042.286</td>
      <td>...</td>
      <td>15469.207</td>
      <td>10.0</td>
      <td>233.07</td>
      <td>8.51</td>
      <td>6.434</td>
      <td>34.635</td>
      <td>60.13</td>
      <td>2.705</td>
      <td>72.58</td>
      <td>0.737</td>
    </tr>
  </tbody>
</table>
<p>427 rows × 59 columns</p>
</div>



##### Let's take a look to the power of the virus.


```python
# calculate number of days
from datetime import date
world_data['date']=pd.to_datetime(world_data['date']).dt.date
d0 = date (2020,1,1)
world_data['no_days']=world_data['date']-d0

# plot the graph
x=world_data['no_days']
y=world_data['total_cases_per_million']

plt.figure(figsize=(10,6))
plt.plot(x,y, label='Total cases per million')
plt.title('Number of COVID-19 Cases per million Worldwide since 2020-01-01')
plt.xlabel('Number of Days since 2020-01-01')
plt.ylabel('Number of Cases per Million')
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,0), useOffset=True)

plt.show()
```

    C:\Users\User\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    C:\Users\User\Anaconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """
    


![png](output_6_1.png)


##### We could see that the number of COVID-19 cases increses rapidly since the first day of the outbreak.

##### The slope of the graph goes deeper and deeper as time goes by.

##### In fact, COVID-19 is a highly infectious virus, which can be spread easily by droplets of saliva or discharge from the nose of infected person.

##### According to World Health Organisation, older people and those with underlying medical problems like diabetes and chronic respiratory disease tend to develop serious illness if infected.

##### Next, we could look at the data of people fighting the pandemic.

##### Number of vaccinations worldwide:


```python
world_data['date']=pd.to_datetime(world_data['date']).dt.date
d0 = date (2020,1,1)
world_data['no_days']=world_data['date']-d0

# plot the graph
x=world_data['no_days']
y=world_data['total_vaccinations']
plt.figure(figsize=(10,6))
plt.plot(x,y, label='Total vaccinations')
plt.title('Number of Total Vaccinaitons Worldwide since 2020-01-01')
plt.xlabel('Number of Days since 2020-01-01')
plt.ylabel('Number of Vaccinations')
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,0), useOffset=True)

plt.show()
```

##### We can see that the number of vaccinations around the world increases steadily.

##### Next, I would like to investigate the relation between the level of development and the resilience of a place. 

##### I assume that wealthier places would have better resilience to the pandemic as they have more medical resources.

##### We could firstly see whether a wealthier place will have lower casualties:


```python
# extract the columns needed for the investigation
df2=df[['location', 'gdp_per_capita', 'total_cases_per_million']]
df2.dropna(inplace=True)

# create a table for plotting graph
gdp_cases=df2.groupby(['location', 'gdp_per_capita'])[['total_cases_per_million']].mean().reset_index()
gdp_cases

# input the content and data 
fig,ax=plt.subplots(figsize=(14,8))
ax=sns.regplot(x='gdp_per_capita', y='total_cases_per_million', data=gdp_cases)
# input the size and frame of the graph
ax.set_title('Relation between GDP per capital to total cases per million')
ax.set_xlabel('GDP per capita')
ax.set_ylabel('Total Cases per Million')

plt.show()
```

    C:\Users\User\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    


![png](output_15_1.png)


##### Secondly, we could investigate whether a wealthier place will have better medical support.


```python
# extract the columns needed for the investigation
df3=df[['location', 'gdp_per_capita', 'handwashing_facilities', 'hospital_beds_per_thousand']]
df3.dropna(inplace=True)

# create a table for plotting graph
handwash=df3.groupby(['location', 'gdp_per_capita'])[['handwashing_facilities']].mean().reset_index()
bed=df3.groupby(['location', 'gdp_per_capita'])[['hospital_beds_per_thousand']].mean().reset_index()

# input the content and data 
x1=handwash['gdp_per_capita']
x2=bed['gdp_per_capita']
y1=handwash['handwashing_facilities']
y2=bed['hospital_beds_per_thousand']

# plot the graph with regression line
fig=plt.figure(figsize=(14,8))
ax1=fig.add_subplot(121)
ax1=sns.regplot(x1, y1)
ax1.set_title('Relation between number of Handwash Facilities and GDP per capital')
ax1.set_xlabel('GDP per capital')
ax1.set_ylabel('Number of handwash facilities')

ax2=fig.add_subplot(122)
ax2=sns.regplot(x2, y2)
ax2.set_title('Relation between Number of Hospital Beds and GDP per capital')
ax2.set_xlabel('GDP per capital')
ax2.set_ylabel('Number of hospital beds')

plt.show()
```

    C:\Users\User\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    


![png](output_17_1.png)


##### These graphs show that even though wealthier places will have more medical support, it does not help with controlling the COVID-19 cases

##### This may be due to a wealthier place tend to hold more people, resulting a higher population density.

##### In order to test my assumption, a graph comparing GDP per capital and population density is created:


```python
#create a table for population density
density=df[['location', 'gdp_per_capita', 'population_density']]
density.dropna(inplace=True)
```

    C:\Users\User\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    


```python
# create a table for population density mean for different countries
density_df=density.groupby(['location', 'gdp_per_capita'])[['population_density']].mean().reset_index()

# input the content and data 
fig,ax=plt.subplots(figsize=(14,8))
ax=sns.regplot(x='gdp_per_capita', y='population_density', data=density_df)
# input the size and frame of the graph
ax.set_title('Relation between GDP per capital to Population Density')
ax.set_xlabel('GDP per capita')
ax.set_ylabel('Population Density')

plt.show()
```


![png](output_21_0.png)


##### The above graph shows that population density has a positive relation with GDP per capital.

##### That's why wealthier places will have more COVID-19 cases even they are richer in medical resources. Not to mention this is a new virus and poeple are lack of experience in fighting it.

##### Hence, no matter you are rich or poor, white or black, male or female, it is really important for us to stand together and fight against the pandemic. It can simply be done by wearing a mask or donate some masks to the poor.

##### It is believed that once we hold our hands together, we could get back to the world before the pandemic, and we could see each other's smile in person.


```python

```
