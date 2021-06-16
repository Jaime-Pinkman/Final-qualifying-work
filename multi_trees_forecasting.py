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


estimator = MultiOutputRegressor(CatBoostRegressor(iterations=100, random_seed=42, loss_function='RMSE', logging_level='Verbose'))
estimator.fit(X_train,y_train)
predictions = estimator.predict(X_test)

plt.plot(y_test[:,0])
plt.plot(predictions[:,0])
print(root_mean_squared_error(y_test[:,0], predictions[:,0]))
