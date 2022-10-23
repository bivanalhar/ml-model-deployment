import numpy as np
import pandas as pd
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def build_dataset(data, label, feature_names):
	if len(data) != len(label):
		print("invalid dataset: quantity of data and label mismatched")
		sys.exit(0)

	label_tp = np.array([label]).transpose()
	dataframe = np.concatenate((data, label_tp), axis = 1)

	cols = feature_names.copy()
	cols.append("value")

	final_data = pd.DataFrame(dataframe, columns = cols)
	return final_data

def eval_metric(actual, prediction):
	rmse = np.sqrt(mean_squared_error(actual, prediction))
	mae = mean_absolute_error(actual, prediction)
	r2 = r2_score(actual, prediction)

	return rmse, mae, r2

def split_dataset(data, label_name = "value", test_size = 0.2):
	train, test = train_test_split(data, test_size = test_size)

	train_data = train.drop([label_name], axis = 1)
	train_label = train[[label_name]]

	test_data = test.drop([label_name], axis = 1)
	test_label = test[[label_name]]

	return train_data, train_label, test_data, test_label