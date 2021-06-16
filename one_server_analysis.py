from functions import *

df, target = get_dataset()
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.33, random_state=42)

mean_value = np.mean(y_train)
print(mean_value)
y_pred_mean = [mean_value for _ in range(len(y_test))]
mean_squared_error(y_test, y_pred_mean, squared=False)

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
