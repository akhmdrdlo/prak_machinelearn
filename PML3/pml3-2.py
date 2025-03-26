# %%
import pandas as pd

# %%
# reading the database
data = pd.read_csv("students.csv")

# %%
# printing the top 10 rows
display(data.head(10))

# %%
# Show the dataset info
data.info()

# %%
# Missing Value Checking
data.isna().sum()

# %%
# Filling Nan Value with mode
data[‘lunch’].fillna(data[‘lunch’].mode()[0], inplace=True)


#%%
# Filling Nan Value with mean
data[‘reading score’].fillna(data[‘reading score’].mean(), inplace=True)


# %%
# Filling Nan Value with mode
data[‘grade’].fillna(data[‘grade’].median()[0], inplace=True)


# %%
# Check the information of the dataset
data.info()

# %%
# Interpolate the Missing Value
data[‘nama_fitur’].interpolate(methode=’linear’, inplace=True)


# %%
#backwardfill the missing value
data[‘’nama_fitur].filna(method= ‘bfill’, inplace = True)

# %%
#forwardfill the missing value
data[‘’nama_fitur].filna(method= ‘bfill’, inplace = True)


# %%
# drop Nan Value
data[‘nama_fitur’].dropna(axis=0, inplace=True)

# drop feature with more than 50% are Nan value
data[‘nama_fitur’].drop(axis=1, inplace=True)
