# %%
# loading libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
import pickle

# %%
df = pd.read_csv('AB_NYC_2019.csv')

# %%
df.drop(['host_id','latitude','longitude','neighbourhood','number_of_reviews','reviews_per_month','id', 'name', 'host_name', 'last_review'], axis=1, inplace=True)


# %%
def Encode(df):
    for column in df.columns[df.columns.isin(['neighbourhood_group', 'room_type'])]:
        df[column] = df[column].factorize()[0]

df_new = Encode(df)

# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_full_train = (df_full_train.price).values
y_test = (df_test.price).values


y_full_train = np.log1p(df_full_train.price.values)
y_test = np.log1p(df_test.price.values)

del df_full_train['price']
del df_test['price']


# %%
full_train_dict = df_full_train.to_dict(orient='records')
test_dict = df_test.to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_full_train = dv.fit_transform(full_train_dict)
X_test = dv.transform(test_dict)

# %%
features = list(dv.get_feature_names_out())

# %%
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

# %%
#%%capture output

xgb_params = {
    'eta': 0.1, 
    'max_depth': 6,
    'min_child_weight': 10,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}
 

# %%
model = xgb.train(xgb_params, dfulltrain, num_boost_round=100)
y_pred = model.predict(dtest)
print(root_mean_squared_error(y_test, y_pred))

# %%
output_filename = 'model.bin'
f_out = open(output_filename, 'wb') # w is for write and b is for binary
pickle.dump((dv, model), f_out)
f_out.close()


