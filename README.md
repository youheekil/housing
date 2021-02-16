# Housing Prediction 

## 1. Boston Housing 

### Miscellaneous Details
* **crim** - per capita crime rate by town
* **zn** - proportion of residential land zoned for lots over 25,000 sq.ft.
* **indus** - proportion of non-retail business acres per town.
* **chas** - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
* **nox** - nitric oxides concentration (parts per 10 million)
RM - average number of rooms per dwelling
* **age** - proportion of owner-occupied units built prior to 1940
* **dis** - weighted distances to five Boston employment centres
* **rad** - index of accessibility to radial highways
* **tax** - full-value property-tax rate per $$10,000$
* **ptratio** - pupil-teacher ratio by town
* **b** - The proportion of blacks by town (1000(Bk - 0.63)^2)
* **lstat** - The proportion of lower status of the population
* **medv** - Median value of owner-occupied homes in $1000's


### 1. Feature Selection
Let's do 'Feature Selection' based on the correlation and covariance. Let's check the correlation between the housing price and the listed hypothesises with graph. The results of 'correlation' and 'covariance' indicates there is relationship between  housing price and other factors.

**Hypothesis 1 - Crime hot spots affect housing price.**
>It is pretty common sense that if crime rate is higher, then housing price is lower. But, sometimes, real data shows different cases, therefore it is always important to chck with the real data.



**Hypothesis 2 - The number of room affects housing price**
>It is pretty common things to happened if the number of room increases, the housing price increases.

#### Feature Extraction :smile:

#### Dimension Reduction - `PCA` and `Regularization`

#### Clustering - `Kmeans`

---
## 2. California Housing
