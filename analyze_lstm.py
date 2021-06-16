from functions import *


df = pd.read_csv('drive/MyDrive/storage/train.csv')
#df.dropna(inplace=True)
#df.fillna(0, inplace=True)

#df = df[[*columns]]
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
dfs_resampled = dfs.resample('3T').mean()

lag = 30
step = 5
df_con, target = get_dataset_concat(dfs_resampled, 'a', lag, step)

diff_con = difference(df_con, 1)
diff_target = difference(target, 1)

scaler_data = MinMaxScaler()
diff_con_scaled = scaler_data.fit_transform(diff_con)

scaler_target = MinMaxScaler()
diff_target_scaled = scaler_target.fit_transform(diff_target)

diff_con_scaled_reshaped = diff_con_scaled.reshape(diff_con_scaled.shape[0], int(diff_con_scaled.shape[1]/lag), lag)
diff_target_scaled_reshaped = diff_target_scaled.reshape(diff_target_scaled.shape[0], int(diff_target_scaled.shape[1]/step), step)

X_train, X_test, y_train, y_test = train_test_split(diff_con_scaled_reshaped, diff_target_scaled_reshaped, test_size=0.33, shuffle=False)

model = Sequential()
model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dense(256))
model.add(Dropout(0.1))
model.add(Dense(128))
model.add(Dropout(0.1))
model.add(Dense(5))
model.compile(loss = root_mean_squared_error, optimizer='adam')

history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

history_1 = eat_and_produce(model, X_test)
inversed_history = scaler_target.inverse_transform(history_1)
y_test_inverted = scaler_target.inverse_transform(y_test.reshape(y_test.shape[0], 5))

target_values = target.values
inverted_target = [inverse_difference(target_values[i], inversed_history[i]) for i in range(len(inversed_history))]
inverted_target_labels = [inverse_difference(target_values[i], y_test_inverted[i]) for i in range(len(y_test_inverted))]

inverted_target_array = np.array(inverted_target)
inverted_target_labels_array = np.array(inverted_target_labels)

plt.plot(inverted_target_array[:, 0], color='orange')
# using shift
plt.plot(needed.values[:, 0], color='b')
print(root_mean_squared_error(needed.values[:, 0], inverted_target_array[:, 0]))
