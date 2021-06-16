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

model = AdaBoostRegressor(learning_rate=0.01, n_estimators=100)
model.fit(X_train, y_train)
predicitons = model.predict(X_test)
rmse = mean_squared_error(y_test, predicitons, squared=False)
print(rmse)

model = CatBoostRegressor(iterations=1000, random_seed=42, loss_function='RMSE', logging_level='Verbose')
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
     logging_level='Verbose',  # you can uncomment this for text output
    plot=True
);

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(rmse)
