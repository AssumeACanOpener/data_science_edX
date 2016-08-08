#!/usr/bin/python3

import pandas as pd
from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn import manifold
import random
import math

# Look pretty...
matplotlib.style.use('ggplot')

#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
# .. your code here .. 
samples = []
colors = []
images = []

#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
# .. your code here .. 
for f in os.listdir('Datasets/ALOI/32/'):
  samples.append(misc.imread('Datasets/ALOI/32/' + f).ravel())
  images.append(misc.imread('Datasets/ALOI/32/' + f))
  colors.append('b')

#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 
for f in os.listdir('Datasets/ALOI/32i/'):
  samples.append(misc.imread('Datasets/ALOI/32i/' + f).ravel())
  images.append(misc.imread('Datasets/ALOI/32i/' + f))
  colors.append('r')

#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 
df = pd.DataFrame(samples)

#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 
iso = manifold.Isomap(n_neighbors=6, n_components=3)
iso.fit(df)
manifold = iso.transform(df)

#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 
num_images, _ = df.shape
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('2D Isomap')
ax.set_xlabel('Component 0')
ax.set_ylabel('Component 1')
x_size = (max(manifold[:,0]) - min(manifold[:,0])) * 0.08
y_size = (max(manifold[:,1]) - min(manifold[:,1])) * 0.08
for img_num in range(0, num_images, 4):
  x0, y0 = manifold[img_num,0]-x_size/2., manifold[img_num,1]-y_size/2.
  x1, y1 = manifold[img_num,0]+x_size/2., manifold[img_num,1]+y_size/2.
  img = images[img_num]
  #ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))
ax.scatter(manifold[:, 0], manifold[:, 1], marker='o', alpha=0.7, c=colors)

#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D Isomap')
ax.set_xlabel('Component 0')
ax.set_ylabel('Component 1')
ax.set_zlabel('Component 2')
ax.scatter(manifold[:, 0], manifold[:, 1], manifold[:, 2], marker='o', alpha=0.7, c=colors)

plt.show()
