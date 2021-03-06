#!/usr/bin/python3

import pandas as pd

# TODO: Load up the 'tutorial.csv' dataset
#
# .. your code here ..
df = pd.read_csv('Datasets/tutorial.csv')


# TODO: Print the results of the .describe() method
#
# .. your code here ..
print('Pandas describe method:')
print(df.describe())


# TODO: Figure out which indexing method you need to
# use in order to index your dataframe with: [2:4,'col3']
# And print the results
#
# .. your code here ..
print('')
print('df.loc[2:4, \'col3\']')
print(df.loc[2:4, 'col3'])
#print(df.loc[df.col1 < 0, 'col3'])
