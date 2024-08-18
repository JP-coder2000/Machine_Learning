"""
In this file I will create the ETL for my project, I'm using the data set of automovile which I retrieved from Kagle.
For more info on the dataset, I addded the files into the repossitory, you can find them in the folder called "Automobile".
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

columns = ["Symboling","Normalized-losses","Make","Fuel-type", "Aspiration", "Num-of-doors", "Body-style","Drive-wheels","Engine-location","Wheel-base","Length","Width","Height", "Curb-weight", "Engine-type" , "Num-of-cylinders", "Engine-size", "Fuel-system", "Bore", "Stroke", "Compression-ratio", "Horsepower", "Peak-rpm", "City-mpg", "Highway-mpg", "Price"]
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
#df.fillna(df.mean(), inplace=True)
#print(df.head())

# Okey, I encountered a problem, the columns "Num-of-doors" and "Num-of-cylinders" are not numerical, so I can't calculate the mean of them.
# The easiest solution is to drop the rows with missing values in these columns. But we are currently seeing in the classroom that I can convert
# columns with string values to numerical values, so I will do that. And after that I will calculate the Mean of the columns.
# I will actually do that for all the columns that are not numerical.

#Creating the dictionaries for the columns that are not numerical
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

# Changing the values of the columns
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

#print(df.dtypes)

#After changing the values, I encountered a new error which was that some columns where "object" type, so I will convert them to numerical values
# in order to calculate the mean of the columns. And finally be able to have all my data in a numerical format.
object_columns = df.select_dtypes(include=['object']).columns
print("Columns with object dtype:", object_columns)

# I got this from this link: https://stackoverflow.com/questions/15891038/change-column-type-in-pandas, very helpful
for col in object_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# Finally I will calculate the mean of the columns and fill the missing values with the mean of the column.
df.fillna(df.mean(), inplace=True)

# Check for any remaining NaN values
print(df.isnull().sum())

# now I will save the data into a new csv file, just because I want to have a backup of the data.
df.to_csv('automobile_cleaned.csv', index = False)
# I only runned it once, so I commented it out to avoid overwriting the file.
# UPDATE I actually made some changes so I will not use this line, I will save it later.

# Now, I needed to verify that the data is not in order.
# I dont see a need to shuffle the data since it it is not a classification problem.

# Now I also need to do one last thing we learned today in class which is to normalize the data.
# I was genuinely searching online for a way to do this without using the sklearn library, but I couldn't find a way to do it.

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Save the scaled data
df_scaled.to_csv('automobile_cleaned_scaled.csv', index=False)

# UPDATE. 
# I believe that the mean data I saved earlier is not the best way to represent the data that is measing, so I will use the mode
# to fill the missing values and save the data again.

# UPDATE.
# It actually didn't work, so I will use the mean to fill the missing values and save the data again. 
# I believe that the mean is the best way to fill the missing values.