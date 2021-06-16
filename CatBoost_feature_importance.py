from functions import *

df = pd.read_csv('drive/MyDrive/storage/train.csv')
#df.dropna(inplace=True)
#df.fillna(0, inplace=True)

dfs = split_servers(df)
format = '%Y-%m-%d %H:%M:%S'
index = pd.to_datetime(df.iloc[0::7, :]['sample_time'], format=format).values#df.iloc[0::7, :]['sample_time']
dfs.reset_index(drop=True, inplace=True)
dfs = dfs.set_index(pd.DatetimeIndex(index))
dfs = dfs.asfreq('T')
dfs['sample_time'] = pd.to_datetime(dfs['sample_time'], format=format)
dfs['day_of_week'] = dfs['sample_time'].dt.dayofweek
dfs['hour'] = dfs.index.hour
dfs.reset_index(drop=True, inplace=True)
dfs.dropna(inplace=True)
dfs.drop(['sample_time'], axis=1, inplace=True)

target = dfs['a_cpu_01_busy']
dfs.drop(['a_cpu_01_busy'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(dfs, target, test_size=0.33, random_state=42, shuffle=True)

train_pool = Pool(X_train, y_train)

model = CatBoostRegressor(iterations=50, random_seed=42, loss_function='RMSE', logging_level='Verbose').fit(train_pool)
feature_importances = model.get_feature_importance(train_pool)
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))
    
model = CatBoostRegressor(iterations=300, random_seed=42, loss_function='RMSE', logging_level='Verbose').fit(train_pool)
object_importances = model.get_object_importance(pool,
                      train_pool,
                      top_size=10,
                      type='Average',
                      update_method='SinglePoint',
                      importance_values_sign='All',
                      thread_count=-1,
                      verbose=True)
