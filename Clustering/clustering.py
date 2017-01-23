import sys
print(sys.version)

from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
#Remove any header by setting the header to None
df = pd.read_csv('dota2Train.csv',header=None)
targets = df.ix[:,0]
print(targets)
df_1 = df.drop(df.columns[[0]], axis=1)
X = np.array(df_1)
#What kind of data is in the rows
n_digits = len(np.unique(targets))
labels = targets
est = KMeans(n_clusters=2, random_state=0).fit(X)
reduced_data = PCA(n_components=2).fit_transform(X)
reduced_data_3 = PCA(n_components=3).fit_transform(X)
fig = plt.figure()
def indic(data):
    #alternatively you can calulate any other indicators
    max = np.max(data, axis=1)
    min = np.min(data, axis=1)
    return max, min
fig = plt.figure()
x,y = indic(reduced_data)
plt.scatter(x, y, marker='x')
fig.savefig("kmeans.png")



from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import random


fig_1 = pylab.figure()
ax = Axes3D(fig_1)

ax.scatter(reduced_data_3[:,0], reduced_data_3[:,1], reduced_data_3[:,2])
pyplot.show()
fig_1.savefig("km2eans.png")

# reduced_data = PCA(n_components=10).fit_transform(X)
# kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
# kmeans.fit(reduced_data)

# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)

# fig = plt.figure()
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#           cmap=plt.cm.Paired,
#           aspect='auto', origin='lower')

# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the targets dataset (PCA-reduced data)')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# fig.savefig("kmeans.png")
plt.show()
print(n_digits)