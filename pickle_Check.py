import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


hd = pd.read_csv(r"C:\Users\ASUS\Downloads\heart.csv")

print(hd.shape)
print(hd.sample(5))

x=hd.drop(columns=['target'])
y=hd['target']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)



lr = RandomForestClassifier()
lr.fit(x_train, y_train)
y_pred = lr .predict(x_test)
print(accuracy_score(y_test, y_pred))

#print(lr.predict([[63,0,2,135,252,0,0,172,0,0,2,0,2]]))

filename='piyush2.pkl'
pickle.dump(lr , open(filename,'wb'))

local_model=pickle.load(open(filename,"rb"))
r=local_model.score(x_test,y_test)
print(r)
print(np.isnan(303))

print(hd.isnull().sum())