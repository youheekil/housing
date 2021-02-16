
## California Housing check

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/youheekil/Desktop/projects/housing/california_housing/california_housing_train.csv")
data.head()
data.columns


# Checking first hypothesis : Median Housing Value descreases as year of housing increases

# Let's check the correlation between the housing price and crime rate with graph
sns.jointplot(x = "housing_median_age" , y ="median_house_value", data = data, kind='reg')

# check 'correlation'
data['housing_median_age'].corr(data['median_house_value'])
# check 'covariance'
data['housing_median_age'].cov(data['median_house_value'])




# Checking second hypothesis : Median Housing Value increases as total_bedrooms increases

# Let's check the correlation between the housing price and number of room number with graph
sns.jointplot(data= data, x ='total_bedrooms', y='median_house_value', kind ='reg')
# check correlation
data['total_bedrooms'].corr(data['median_house_value'])
# check covariance
data['total_bedrooms'].cov(data['median_house_value'])


## according to the result of correlation and covariance, the relationship
## between`total_bedrooms` and `median_house_value`are having little positive relationship.



# Let's create a new variable by using 'PCA'
corr_bar = []
for column in data.columns:
    print(f'{column} and median_house_value correlation: {data["median_house_value"].corr(data[column])}\n')
    corr_bar.append(data["median_house_value"].corr(data[column]))


sorted(corr_bar, reverse=False)
# median_income and median_house_value has the highest correlation and
map(abs, corr_bar)
sns.barplot(data.columns, corr_bar)

# variable 'd' and 'dis' has low correlation
x = data[['housing_median_age', 'total_rooms']]

# call library for 'PCA'
from sklearn.decomposition import PCA

# create 1 random variable by combing 2 random variable
pca = PCA(n_components = 1)

# let data learn
pca.fit(x)

# check covariance of each rv in new random variable
pca.components_

# pc1's explanable covariance
pca.explained_variance_ratio_

# add this new random variable into new column called 'pc1'
data['pc1'] = pca.fit_transform(x)

# check correlation again with new random variable
sns.jointplot(data= data, x = 'pc1', y ='median_house_value', kind= 'reg')

# check correlation with new random variable
data["pc1"].corr(data['median_house_value'])



#=========================================================================##                             Normalization                               #
#=========================================================================#

# library for Normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# normlize x after training x
scaler.fit(x)
scaler_x = scaler.transform(x)

## repeated work

# find number of pca
pca = PCA(n_components=1)

# train data
pca.fit(scaler_x)

# check each new variable's covariance, then you can find different covariance compared to result above
pca.components_

# explained_variance_ratio_ of new variable, 'pc1'
pca.explained_variance_ratio_

data['pc1'] = pca.fit_transform(scaler_x)
sns.jointplot(data = data, x ='pc1', y = 'median_house_value', kind = 'reg')
data['pc1'].corr(data['median_house_value'])

# looks like it's closer to 0 after regualization

#=========================================================================##                             CLustering                                   #
#=========================================================================#

# Let's look for groups with similar tendencies through clustering.
data.head()

median_house_value =  data['median_house_value']
del data['median_house_value']
