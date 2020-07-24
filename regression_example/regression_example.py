# example from chapter 2 of the book
# practice with univariate regression
# dataset based on 1990 California Census - predict a district's median housing price
import os
import tarfile
import urllib.request
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# fetching the data
DOWNLOAD_ROUTE = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROUTE + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# loading data to pandas
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# fetch_housing_data() //uncomment to load the tgz file
# pandas dataset
housing = load_housing_data()
# housing.info() will show description of data and data types
# houseing.describe() will show summary of statistical attributes

# using matplot lib to study the dataset
#housing.hist(bins=50, figsize=(20,15))
#plt.show()

# creating test set - 80% training 20% testing

# how a hard coded function would look like
# iloc selects rows
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
# train_set, test_set = split_train_test(housing, 0.2)
# use sklearn to efficienlty split training and testing
# train_set, test_set = train_test_split(housinng, test_size=0.2, random_state=42)

# categorize the incomes
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1,2,3,4,5])
# stratified sampling based on income category
# why use this instead of normally splitting up the data?
# when looking at the medium incomes they arent represented equally so the test and training set must be proprtional to that
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# remove the income_cat attribute so the data is back to original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# ***Visualizing the Training Set***
# create a copy of the training set so the original isn't changed
housing = strat_train_set.copy()
# scatter plot that georgraphically shows the housing prices in california
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, 
    s=housing["population"]/100, label = "population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True
)
plt.legend()
# use plt.show() to see the plot

# computing the Standard Correlation Coefficient for each feature
# closer to postive 1 means stronger correlation, close to 0 no correlation, -1 negative correlation
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# print above statement to view the values
# ***ANOTHER WAY****
# plotting the scatter plot of each feature using Panda's scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))

# transform tail heavy features to be more normal (take the log of it)
# ***Attribute Combinations***
# in this case total number of rooms is not useful if you dont know how many houses there are
# total number of bedrooms not useful without the number of rooms
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]
# now looking at the correlations with the new features
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# print the above statement to see the values


