# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 

2.Set variables for assigning dataset values. OD

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S.Rohith kumar
RegisterNumber:  24004371
*/
```
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv('student_scores.csv') 

print(df) 

df.head(0)

df.tail(0)

print(df.head()) 

print(df.tail()) 

x=df.iloc[:,:-1].values 

print(x) 

y=df.iloc[:,1].values 

print(y) 

from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression 

regressor = LinearRegression() 

regressor.fit(x_train,y_train) 

y_pred = regressor.predict(x_test) 

print(y_pred) 

print(y_test) 

#Graph plot for training data

plt.scatter(x_train,y_train,color='black')

plt.plot(x_train,regressor.predict(x_train),color="blue")

plt.title("Hours vsScores(Training set)") 

plt.xlabel("Hours") 

plt.ylabel("Scores")

plt.show()

#Graph plot for test data 

plt.scatter(x_test,y_test,color='black')

plt.plot(x_train, regressor.predict(x_train), color="red") 

plt.title("Hours vs Scores(Testing set)") 

plt.xlabel("Hours")

plt.ylabel("Scores") 

plt.show()

mse=mean_absolute_error(y_test, y_pred)

print('MSE = ',mse) 

mae=mean_absolute_error(y_test, y_pred) 

print('"MAE = ',mae)

rmse=np.sqrt(mse) 

print("RMSE= ",rmse)


## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![Screenshot (10)](https://github.com/user-attachments/assets/f1ac85f1-3cbe-4d2f-83ab-7013b02255ea)
![Screenshot (11)](https://github.com/user-attachments/assets/a0f8a0ef-1302-44d9-8cbe-42dadaaef2a8)
![Screenshot (12)](https://github.com/user-attachments/assets/67a4313e-1ab5-4548-b9f5-bce9c41b642f)
![Screenshot (13)](https://github.com/user-attachments/assets/8b61698f-d8b2-40b1-8910-f5f70d6c6082)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
