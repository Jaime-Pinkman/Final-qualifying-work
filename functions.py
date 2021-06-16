import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime, time
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from catboost import Pool, CatBoostRegressor, metrics, cv
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, Conv1D, Bidirectional
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
from sklearn.preprocessing import MinMaxScaler

def get_dataset():
    df = pd.read_csv('drive/MyDrive/train.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    df["m_id"] = df["m_id"].astype('category')
    df["m_id"] = df["m_id"].cat.codes
    target = df['cpu_01_busy']
    df.drop(['cpu_01_busy', 'sample_time'], axis=1, inplace=True)
    return df, target

def split_servers(df_local):
	df_local = df_local.copy()
	servers = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
	df_server = {}
	for i in range(len(servers)):
		df_server[servers[i]] = df_local.iloc[i::len(servers), :]
		df_server[servers[i]].reset_index(drop=True, inplace=True)
		df_server[servers[i]] = df_server[servers[i]].add_prefix(servers[i] + '_')
		df_server[servers[i]].drop([servers[i] + '_'+'m_id'], axis=1, inplace=True)
		df_server[servers[i]].rename(columns={servers[i] + '_'+'sample_time': 'sample_time'}, inplace=True)
	df_concat = pd.concat(list(df_server.values()), axis=1)
	df_concat = df_concat.loc[:,~df_concat.columns.duplicated()]
	return df_concat
    
def series_to_supervised(data, columns, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('%s(t-%d)' % (columns[j], i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('%s(t)' % (columns[j])) for j in range(n_vars)]
		else:
			names += [('%s(t+%d)' % (columns[j], i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
  
def get_dataset_concat(df_local, server, n_in=1, n_out=1):
	df_local = df_local.copy()
	#df_local.drop(['sample_time'], axis=1, inplace=True)
	values = df_local.values

	data = series_to_supervised(values, df_local.keys(), n_in, n_out)
	target_0 = data[server+'_'+'cpu_01_busy'+'(t)']
	target_1 = data[server+'_'+'cpu_01_busy'+'(t+1)']
	target_2 = data[server+'_'+'cpu_01_busy'+'(t+2)']
	target_3 = data[server+'_'+'cpu_01_busy'+'(t+3)']
	target_4 = data[server+'_'+'cpu_01_busy'+'(t+4)']

	target = pd.concat([target_0, target_1, target_2, target_3, target_4], axis = 1)#, target_3, target_4]

	data = data.loc[:,~data.columns.str.endswith('(t)')]
	data = data.loc[:,~data.columns.str.endswith('(t+1)')]
	data = data.loc[:,~data.columns.str.endswith('(t+2)')]
	data = data.loc[:,~data.columns.str.endswith('(t+3)')]
	data = data.loc[:,~data.columns.str.endswith('(t+4)')]
	return data, target

def difference(dataset, interval=1):
	diff = list()
	dataset = dataset.values.copy()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# invert differenced forecast
def inverse_difference(last_ob, value):
	return value + last_ob
  
def eat_and_produce(model, data):
	preds = model.predict(data)
	print(np.shape(preds))
	return preds

def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true)))
