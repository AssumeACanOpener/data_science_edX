import pandas as pd
import numpy as np


# TODO:
# Load up the dataset, setting correct header labels
# Use basic pandas commands to look through the dataset...
# get a feel for it before proceeding!
# Find out what value the dataset creators used to
# represent "nan" and ensure it's properly encoded as np.nan
#
# .. your code here ..
df = pd.read_csv('Datasets/census.data', names = ['education', 'age', 'capital-gain', 'race', 'capital-loss', 'hours-per-week', 'sex', 'classification'], index_col = 0)

# TODO:
# Figure out which features should be continuous + numeric
# Conert these to the appropriate data type as needed,
# that is, float64 or int64
#
# .. your code here ..
df.age = pd.to_numeric(df.age, errors='coerce')
df['capital-gain'] = pd.to_numeric(df['capital-gain'], errors='coerce')
df['capital-loss'] = pd.to_numeric(df['capital-loss'], errors='coerce')
df['hours-per-week'] = pd.to_numeric(df['hours-per-week'], errors='coerce')

# TODO:
# Look through your data and identify any potential categorical
# features. Ensure you properly encode any ordinal types using
# the method discussed in the chapter.
#
# .. your code here ..
edu_levels = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Doctorate']
df.education = df.education.astype('category', ordered = True, categories = edu_levels).cat.codes
classification_levels = ['<=50K', '>50K']
df.classification = df.classification.astype('category', ordered = True, categories = classification_levels).cat.codes

# TODO:
# Look through your data and identify any potential categorical
# features. Ensure you properly encode any nominal types by
# exploding them out to new, separate, boolean fatures.
#
# .. your code here ..
df = pd.get_dummies(df, columns=['race'])
df = pd.get_dummies(df, columns=['sex'])

# TODO:
# Print out your dataframe
