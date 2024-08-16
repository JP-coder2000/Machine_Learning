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

# After taking a big consideration ( I read the documentation) I decided to drop the first column because it is not relevant (I'm ASSUMING) and 
# because it has a lot of missing values.
# The oder columns with missing values I will do what Benji told us about inputing the missing values with the mean of the column.

df.drop("Normalized-losses", axis = 1, inplace = True)
df.fillna(df.mean(), inplace=True)
print(df.head())

# Okey, I encountered a problem, the columns "Num-of-doors" and "Num-of-cylinders" are not numerical, so I can't calculate the mean of them.
# The easiest solution is to drop the rows with missing values in these columns. But we are currently seeing in the classroom that I can convert
# columns with string values to numerical values, so I will do that. And after that I will calculate the Mean of the columns.
# I will actually do that for all the columns that are not numerical.

# Crear diccionarios para convertir valores de texto a valores numéricos
make_dict = {
    "alfa-romero": 1, "audi": 2, "bmw": 3, "chevrolet": 4, "dodge": 5, "honda": 6, "isuzu": 7, "jaguar": 8,
    "mazda": 9, "mercedes-benz": 10, "mercury": 11, "mitsubishi": 12, "nissan": 13, "peugot": 14, "plymouth": 15,
    "porsche": 16, "renault": 17, "saab": 18, "subaru": 19, "toyota": 20, "volkswagen": 21, "volvo": 22
}

fuel_type_dict = {"diesel": 1, "gas": 2}

aspiration_dict = {"std": 1, "turbo": 2}

num_of_doors_dict = {"two": 2, "four": 4}

body_style_dict = {"hardtop": 1, "wagon": 2, "sedan": 3, "hatchback": 4, "convertible": 5}

drive_wheels_dict = {"4wd": 1, "fwd": 2, "rwd": 3}

engine_location_dict = {"front": 1, "rear": 2}

engine_type_dict = {"dohc": 1, "dohcv": 2, "l": 3, "ohc": 4, "ohcf": 5, "ohcv": 6, "rotor": 7}

num_of_cylinders_dict = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "twelve": 12}

fuel_system_dict = {"1bbl": 1, "2bbl": 2, "4bbl": 3, "idi": 4, "mfi": 5, "mpfi": 6, "spdi": 7, "spfi": 8}

# Aplicar el mapeo a las columnas correspondientes
df["Make"] = df["Make"].map(make_dict)
df["Fuel-type"] = df["Fuel-type"].map(fuel_type_dict)
df["Aspiration"] = df["Aspiration"].map(aspiration_dict)
df["Num-of-doors"] = df["Num-of-doors"].map(num_of_doors_dict)
df["Body-style"] = df["Body-style"].map(body_style_dict)
df["Drive-wheels"] = df["Drive-wheels"].map(drive_wheels_dict)
df["Engine-location"] = df["Engine-location"].map(engine_location_dict)
df["Engine-type"] = df["Engine-type"].map(engine_type_dict)
df["Num-of-cylinders"] = df["Num-of-cylinders"].map(num_of_cylinders_dict)
df["Fuel-system"] = df["Fuel-system"].map(fuel_system_dict)

# Verificar el resultado
print(df.head())

# Imputar los valores faltantes con la media de la columna
df.fillna(df.mean(), inplace=True)

# Verificar que no haya valores faltantes
print(df.isnull().sum())


