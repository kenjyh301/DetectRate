import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


data= pandas.read_csv("dataset/train.csv")
X= data.loc[:,'Gender':'Burn Rate']
X=X.dropna()
y= X['Burn Rate']
X= X.loc[:,'Gender':'Mental Fatigue Score']

column= X.columns.tolist()
for column_name in column[0:3]:
    X[column_name]=pandas.factorize(X[column_name],sort=True)[0]

X,X_test= train_test_split(X,test_size=0.3)
y,y_test= train_test_split(y,test_size=0.3)

def GradJ(x,y,w1,w0):
    x=np.transpose(x)
    m=x.shape[1]
    h= np.dot(w1,x)+w0
    err= h-y
    gradJ1=np.zeros_like(w1)
    for i in range(m):
        gradJ1+=1/m*err[i]*x[:,i]
    gradJ0=1/m*sum(err)
    # print([gradJ1,gradJ0])
    return [gradJ1,gradJ0]

def SEs(x,y,w1,w0):
    m=x.shape[0]
    h= np.dot(w1,np.transpose(x))+w0
    err= h-y
    J=0.5/m*sum(err**2)
    return J

def Alg(x,y,X_test,y_test):
    w1=np.ones(x.shape[1])
    w0=1
    alpha=0.02
    tolerance=0.01
    numEpoch=0
    maxEpoch=1000
    J=SEs(x,y,w1,w0)

    while abs(J)>tolerance:
        print("Epoch "+str(numEpoch))
        [grad1,grad0]=GradJ(x,y,w1,w0)
        w1=w1-alpha*grad1
        w0=w0-alpha*grad0
        J=SEs(x,y,w1,w0)
        numEpoch+=1
        if(numEpoch>=maxEpoch):
            break
        # print([w1,w0,J,grad1,grad0])
        accuracy=test(X_test,y_test,0.1,w1,w0)
        if(accuracy>=0.97):
            break
    print([w1,w0])
    return [w1,w0,numEpoch]

def test(X,y,tolerance,w1,w0):
    predict= np.dot(w1,np.transpose(X))+w0
    error= y-predict
    numSample=error.size
    numCorrect=0
    for i in range(error.size):
        if abs(error[i])<=tolerance:
            numCorrect+=1
    print("Accuracy "+str(float(numCorrect)/numSample))
    return float(numCorrect)/numSample

def main():
    global X
    global y
    global X_test
    global y_test
    print(X)
    print(y)
    X= pandas.DataFrame.to_numpy(X)
    X_test= pandas.DataFrame.to_numpy(X_test)
    y=y.values
    y_test= y_test.values
    [w1,w0,numEpoch]=Alg(X,y,X_test,y_test)
    test(X_test,y_test,0.1,w1,w0)
    # print(X.shape[0])
    
    

if __name__=="__main__":
    main()
