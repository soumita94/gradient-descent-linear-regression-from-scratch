import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Load the data

df = pd.read_csv('Student_synthetic.csv')

# divide data in x and y

X = df.iloc[:,0:3].values
y = df.iloc[:,3].values

#visualizing each column

# Plot 1: Study hours vs Performance
plt.scatter(df.iloc[:, 0], df.iloc[:, 3])
plt.xlabel("Study Hours")
plt.ylabel("Performance Score")
plt.title("Study Hours vs Performance")
plt.show()

# Plot 2: Sleep hours vs Performance
plt.scatter(df.iloc[:, 1], df.iloc[:, 3])
plt.xlabel("Sleep Hours")
plt.ylabel("Performance Score")
plt.title("Sleep Hours vs Performance")
plt.show()

# Plot 3: Break minutes vs Performance
plt.scatter(df.iloc[:, 2], df.iloc[:, 3])
plt.xlabel("Break Minutes")
plt.ylabel("Performance Score")
plt.title("Break Minutes vs Performance")
plt.show()

#Split the data 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=2)

#fit into Linear Regression
lr= LinearRegression()
lr.fit(X_train,y_train)
print(lr.coef_)
print(lr.intercept_)

# calculate the r2_score

y_pred = lr.predict(X_test)
#print(r2_score(y_test,y_pred))

# Start making batch gradient descent class

class GDR:
    def __init__(self,learning_rate=0.001,epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercpt_ = None
    def fit(self,X_train,y_train):
        #initialize your coef
        self.intercept_=0
        self.coef_=np.ones(X_train.shape[1])
        for i in range(self.epochs):
            #update intercept
            y_hat = np.dot(X_train,self.coef_) + self.intercept_
            intercept_der = -2* np.mean(y_train - y_hat)
            self.intercept_= self.intercept_ - (self.learning_rate * intercept_der)
            #update all coef
            coef_der = -2 * np.dot((y_train - y_hat),X_train)/X_train.shape[0]
            self.coef_ = self.coef_ - (self.learning_rate * coef_der)
        print(self.intercept_,self.coef_)
    def predict(self,X_test):
        return np.dot(X_test,self.coef_)+ self.intercept_
    
gdr=GDR(epochs=30,learning_rate=0.0001)
gdr.fit(X_train,y_train)

y_pred = gdr.predict(X_test)
r2_score(y_test,y_pred)
    




