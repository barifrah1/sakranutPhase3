import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# split a dataset into train and test sets
import pickle


class DataLoader:
    def __init__(self, args, is_grid_search):
        self.args = args
        self.is_grid_search = is_grid_search
        self.data = pd.read_csv(
            self.args['fileName'])  # , nrows=100000
        self.empty = 0

    # split data into 70% train , 15% validation and 15% test
    def split_train_validation_test(self, x, y):
        X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(
            x, y, test_size=0.15)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_and_val, y_train_and_val, test_size=len(X_test))

        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess(self):
        self.data = self.data.loc[self.data['state'].isin(
            ['failed', 'successful'])]
        # drop na values
        self.data = self.data.dropna()
        self.data['main_category'] = 'main_category_' + \
            self.data['main_category']
        self.data = self.data.drop(['ID',
                                    'launched', 'cat_sub_cat', 'log_usd_goal_real'], 1)
        self.data.loc[~self.data['country'].isin(
            ['US', 'GB']), 'country'] = 'OTHER_COUNTRY'
        self.data.loc[~self.data['currency'].isin(
            ['USD', 'GBP', 'EUR', 'CAD', 'AUD']), 'currency'] = 'OTHER_CURRENCY'
        # round to next 10's numbers
        self.data['duration'] = 10*round(self.data['duration']/10)
        self.data['duration'] = self.data['duration'].astype(int)
        self.data = self.data[self.data['duration'] > 5]
        self.data['goal_level'] = 0

        cat_columns = ['state']
        self.data['state'] = self.data['state'].astype('category')
        self.data[cat_columns] = self.data[cat_columns].apply(
            lambda x: x.cat.codes)
        # one hot encoding for categorical variables

        i = 0
        while i <= 0.95:
            if i == 0:
                mean = self.data.loc[self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(
                    i+0.05)]['usd_goal_real'].mean()
                self.data.loc[(self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(
                    i+0.05)), 'goal_level'] = int(mean)
                i += 0.05
                i = round(i, 2)
                continue
            if i == 0.95:
                mean = self.data.loc[self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(
                    i)]['usd_goal_real'].mean()
                self.data.loc[(self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(
                    i)), 'goal_level'] = int(mean)
                i += 0.05
                i = round(i, 2)
                break

            if i == 0.45:
                mean = self.data.loc[((self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(i)) & (
                    self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(i+0.058)))]['usd_goal_real'].mean()
                self.data.loc[((self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(i)) & (
                    self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(i+0.058))), 'goal_level'] = int(mean)
                i += 0.05
                i = round(i, 2)
                continue

            mean = self.data.loc[((self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(i)) & (
                self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(i+0.05)))]['usd_goal_real'].mean()

            self.data.loc[((self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(i)) & (
                self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(i+0.05))), 'goal_level'] = int(mean)
            i += 0.05
            i = round(i, 2)
        columns_to_drop = ['name', 'usd_goal_real', 'state']
        # for data generation
        self.option_dict = {}
        for column in self.data:
            if(column in columns_to_drop):
                continue
            if column not in self.option_dict.keys():
                self.option_dict[column] = []
            try:
                self.option_dict[column] = list(self.data[column].unique())
            except:
                print("fdf")
        # calculate optional values of categories per main category and optional currency per country
        self.category_per_main_category = {}
        self.currency_per_country = {}
        for main_cat in self.option_dict['main_category']:
            self.category_per_main_category[main_cat] = list(
                self.data[self.data['main_category'] == main_cat]['category'].unique())
        for coun in self.option_dict['country']:
            self.currency_per_country[coun] = list(
                self.data[self.data['country'] == coun]['currency'].unique())
        category = pd.get_dummies(self.data['category'], drop_first=False)
        main_category = pd.get_dummies(
            self.data['main_category'], drop_first=True)
        currency = pd.get_dummies(self.data['currency'], drop_first=False)
        country = pd.get_dummies(self.data['country'], drop_first=False)
        self.data = self.data.drop(
            ['category', 'main_category', 'currency', 'country', 'usd_goal_real'], 1)
        self.data = pd.concat(
            [self.data, category, main_category, currency, country], axis=1)
        Y = self.data.loc[:, 'state'].values
        self.data = self.data.drop(['state', 'name'], 1)
        # data scaling
        self.scale_dict = {}
        for column in self.data:
            if column not in self.scale_dict.keys():
                self.scale_dict[column] = [0, 0]
            self.scale_dict[column][0] = self.data[column].mean()
            self.scale_dict[column][1] = self.data[column].std()
            """if column not in self.option_dict.keys():
                self.option_dict[column] = []
            try:
                self.option_dict[column] = list(self.data[column].unique())
            except:
                print("fdf")"""
        with open('scale_dict.pickle', 'wb') as handle:
            pickle.dump(self.scale_dict, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open('option_dict.pickle', 'wb') as handle:
            pickle.dump(self.option_dict, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open('category_per_main_cat_dict.pickle', 'wb') as handle:
            pickle.dump(self.category_per_main_category, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open('currency_per_country_dict.pickle', 'wb') as handle:
            pickle.dump(self.currency_per_country, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        self.empty = self.data.iloc[0:0]
        self.empty.to_pickle("empty_data_format.pkl")
        columns = self.data.columns
        self.data.columns = columns
        self.data = norm(self.data, self.scale_dict)
        features = columns
        X = self.data.values
        return X, Y, features

    # get empty format dataframe and return row ready to be enter to NN
    # x~ [category,main_category,currency,country,goal_level,duration,year_launched,month_launched]


def random_project_preproesses(empty, x, scale_dict):
    print(x)
    empty = empty.iloc[0:0]
    a = np.zeros(shape=(1, len(empty.columns)))
    empty = pd.DataFrame(a, columns=empty.columns)
    empty.loc[0, f'{x[0]}'] = 1
    empty.loc[0, f'{x[1]}'] = 1
    empty.loc[0, f'{x[2]}'] = 1
    empty.loc[0, f'{x[3]}'] = 1
    empty.loc[0, 'goal_level'] = x[4]
    empty.loc[0, 'duration'] = x[5]
    empty.loc[0, 'year_launched'] = x[6]
    empty.loc[0, 'month_launched'] = x[7]
    print('empty.loc[0', empty.loc[0])
    row = norm(empty, scale_dict)
    row = row.values
    return row

# get data x and mean&std dictionary and return the norm data


def norm(x, y):
    for column in x:
        x[column] = (x[column]-y[column][0])/(y[column][1])
    return x
