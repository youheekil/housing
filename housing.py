# Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/youheekil/github/housing/boston.csv")
data.head()

# Checking first hypothesis

# Let's check the correlation between the housing price and crime rate with graph
sns.jointplot(data = data, x = 'crim', y = 'medv', kind = 'reg')
# check 'correlation'
data['crim'].corr(data['medv'])
# check 'covariance'
data['crim'].cov(data['medv'])


# Checking second hypothesis

# Let's check the correlation between the housing price and number of room number with graph
sns.jointplot(data= data, x ='rm', y='medv', kind ='reg')
# check correlation
data['rm'].corr(data['medv'])
# check covariance
data['rm'].cov(data['medv'])

# Let's create a new variable by using 'PCA'

corr_bar = []
for column in data.columns:
    print(f'{column} and medv correlation: {data["medv"].corr(data[column])}\n')
    corr_bar.append(data["medv"].corr(data[column]))

map(abs, corr_bar)
sns.barplot(data.columns, corr_bar)

# variable 'd' and 'dis' has low correlation
x = data[['dis', 'b']]

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
sns.jointplot(data= data, x = 'pc1', y ='medv', kind= 'reg')

# check correlation with new random variable
data["pc1"].corr(data['medv'])

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
sns.jointplot(data = data, x ='pc1', y = 'medv', kind = 'reg')
data['pc1'].corr(data['medv'])


#=========================================================================##                             CLustering                                   #
#=========================================================================#

# Let's look for groups with similar tendencies through clustering

del data['chas'] # remove variable 'chas' where containing categorical data

medv = data['medv']
del data['medv']


# 군집화를 진행하기에 앞서 수월한 시각화 및 설명력을 첨부하기 위하여 PCA를 통해 변수를 2개로 압축해보자.
# 필요 라이브러리를 불러옵니다


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Perform regualization
# Create object (an object can be a variable, a data structure, a function, or a method, and as such, is a value in memory referenced by an identifier)
scaler = StandardScaler()


# Fitting the data
scaler.fit(data)

# transform
scaler_data = scaler.transform(data)

## since we use the same data for fitting and transforming, the result is the same with using "fit_transform" function

scaler_data_fit = scaler.fit_transform(data)

## as we can check from this code, results of the two aboce is the same.
scaler_data == scaler_data_fit

# Create object (PCA)
pca = PCA(n_components = 2)

# fitting PCA
pca.fit(scaler_data)

# Check results - array
pca.transform(scaler_data)

# Transform to Data Frame
data2 = pd.DataFrame(
    data = pca.transform(scaler_data),
    columns = ['pc1', 'pc2']
    )

data2.head()


#========================================================================## #                           kmeans                                        #
#=========================================================================#

from sklearn.cluster import KMeans

x = [] # store number of k
y = [] # store number of inertia_(응집도)

for k in range(1,30):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(data2)

    x.append(k)
    y.append(kmeans.inertia_)


plt.plot(x, y)

# create object
kmeans = KMeans(n_clusters = 4) # elbow point : 4

# fitting data2
kmeans.fit(data2)

data2['labels'] = kmeans.predict(data2)
data2.head()


sns.scatterplot(x= 'pc1', y='pc2', hue = 'labels', data = data2)

## Interpret clustering results

# add the 'medv' back to the data
data2['medv'] = medv

data2.head()



# create a varialbe for a value of medv per group for visualisation purpose

mdedv_0 = data2[data2['labels']==0]['medv'].mean()
mdedv_1 = data2[data2['labels']==1]['medv'].mean()
mdedv_2 = data2[data2['labels']==2]['medv'].mean()
mdedv_3 = data2[data2['labels']==3]['medv'].mean()

sns.barplot(x = ['group_0','group_1','group_2','group_3'],
            y = [mdedv_0, mdedv_1, mdedv_2, mdedv_3])

# highest group is 'group_1' and the lowest group is 'group_2' (but it changes as KNN is doing randomly)

# duplicate labels into original data
data['labels'] = data2['labels']
data.head()

# 각 그룹의 데이터를 나누어서 변수에 담습니다.
group = data[(data['labels']==1) | (data['labels']==2)]

group = group.groupby('labels').mean().reset_index()
group




#plt.subplots(행, 열, figsize=())
f, ax = plt.subplots(2, 6, figsize = (20, 13))

sns.barplot(x='labels', y='crim', data = group, ax = ax[0,0])
sns.barplot(x='labels', y='zn', data = group, ax = ax[0,1])


column = group.columns
f, ax = plt.subplots(2, 6, figsize=(20, 13))

for i in range(1, 12) :
    sns.barplot(x = 'labels', y = column[i], data = group, ax = ax[i//6, i%6])
