# %%
import pandas as pd

# %%
# reading the database
data = pd.read_csv("DatasetForCoffeeSales2.csv")

# %%
# printing the top 10 rows
display(data.head(10))
