from sklearn.naive_bayes import GaussianNB
import pandas as pd 
import numpy as np 

# you can type the data by yourself in python file
# data = {'Gender' : ['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female']
# 	, 'height' : [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75]
# 	, 'weight' : [180, 190, 170, 165, 100, 150, 130, 150]
# 	, 'foot size' : [12, 11, 12, 10, 6, 8, 7, 9]
# 	}

# or you can read data from a csv file
data = pd.read_csv('data.csv')

x = []
feature = data.select(lambda x: x != 'Gender', axis=1)
for index, row in feature.iterrows():
	x.append(list(row))
x = np.array(x)
y = np.array(list(data['Gender']))

model = GaussianNB()

model.fit(x, y)

predicted = model.predict(np.array([6, 130, 8]).reshape(1, -1))
print predicted