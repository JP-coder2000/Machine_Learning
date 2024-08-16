"""
In this file I will create the ETL for my project, I'm using the data set of automovile which I retrieved from Kagle.
For more info on the dataset, I addded the files into the repossitory, you can find them in the folder called "Automobile".
"""


import pandas as pd
import numpy as np

columns = ["Symboling","Normalized-losses","Make","Fuel-type", "Aspiration", "Num-of-doors", "Body-style","Drive-wheels","Wheel-base","Length","Width","Height", "Curb-weight", "Engine-type" , "Num-of-cylinders", "Engine-size", "Fuel-system", "Bore", "Stroke", "Compression-ratio", "Horsepower", "Peak-rpm", "City-mpg", "Highway-mpg", "Price"]
df = pd.read_csv('/Users/juanpablocabreraquiroga/Documents/Machine_Learning/ETL/automobile/imports-85.data', names = columns)

#print(df.head())

# Reading the documentation, I found that the missing values are represented by "?", so I will replace them with NaN because
# it is easier to work with NaN values in pandas

df.replace("?", np.nan, inplace = True)

missing_values_per_column = df.isnull().sum()
print(missing_values_per_column)