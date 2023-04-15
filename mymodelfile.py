from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

'''
Modify following class to develop your model
'''
class MyModel:
	'''
	Initialize the model and its parameters
	'''
	def __init__(self):

		# write code to define your model
		self._model = DummyRegressor(strategy = 'constant',
									  constant = 10**6)

	def fit(self, training_data):

		# create training data
		# create dummy data now
		dummy_training_data = [["MA Chidambaram Stadium",1,"Mumbai Indians","Kolkata Knight Riders",
								"Quinton de Kock, Rohit Sharma, Suryakumar Yadav",
								"Harbhajan Singh, Varun Chakravarthy, Shakib Al Hasan, Pat Cummins"],
								["MA Chidambaram Stadium",2,"Kolkata Knight Riders","Mumbai Indians",
								"Harbhajan Singh, Varun Chakravarthy, Shakib Al Hasan, Pat Cummins",
								"Quinton de Kock, Rohit Sharma, Suryakumar Yadav"]]
		dummy_training_labels = np.array([30,30]).reshape(-1,1)

		training_data = pd.DataFrame(data=dummy_training_data,
									 columns = ["venue", "innings", "batting_team", 
									 			"bowling_team", "batsmen", "bowlers"])

		# train the model
		self._model.fit(dummy_training_data,dummy_training_labels)

		return self
		
	def predict(self, test_data):
		# NOT USED
		X_test = test_data

		# compuate and return predictions
		return  pd.DataFrame({ 'id':[0,1], 'predicted_runs':[40,37]})