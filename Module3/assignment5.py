#!/usr/bin/python3
from pandas.tools.plotting import andrews_curves
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Look pretty...
matplotlib.style.use('ggplot')

df = pd.read_csv('Datasets/wheat.data')
#df = df.drop('id', axis=1)
#df = df.drop('area', axis=1)
#df = df.drop('perimeter', axis=1)

plt.figure()
andrews_curves(df, 'wheat_type', alpha=0.4)
plt.show()
