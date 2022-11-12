import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as datasets

# Data exploration

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

df.columns

df.value_counts()

x = df.iloc[:].values

plt.figure(figsize=(8,6) , dpi=70)
sns.histplot(data =df , x="sepal length (cm)" , bins=20)

# Scaling data for clustering

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scalerx = scaler.fit_transform(x)

from sklearn.cluster import KMeans
model = KMeans( n_clusters =2 )
clusters =model.fit_predict( scalerx ) 

x['cluster'] =clusters
x.head()

#CORRELATING DATA

#tells how each feature is correlated to cluster
x.corr()['cluster']

x.corr()['cluster'] .iloc[:-1]

x.corr()['cluster'] .iloc[:-1].sort_values().plot(kind = 'bar')

#Elbow method

from sklearn.cluster import KMeans
ssd=[]
for i in range( 2 , 11):
    model = KMeans( n_clusters = i , init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    model.fit( scalerx )
    ssd.append( model.inertia_ )
ssd

plt.plot( range(2,11) , ssd , "o--" , color="red")
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Square")
plt.title("ELBOW METHOD")

plt.scatter(df.iloc[:,2],df.iloc[:,3],c = clusters,cmap = 'viridis')
plt.scatter(model.cluster_centers_[:,2],model.cluster_centers_[:,3],c='red',marker = 'x')
