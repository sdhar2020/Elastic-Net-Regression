import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

def read_sort(file):
    df = pd.read_csv('sphist.csv')
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
    df.sort_values('Date', axis=0, ascending=True, inplace=True)
    cols= df.columns.tolist()
    return(df,cols)

def rolling_stats(df,col,w):
    df['{0}_mean_{1}'.format(col, w)]= df.loc[:,col].rolling(window=w).mean().shift(1)
    df['{0}_std_{1}'.format(col,w)]= df.loc[:,col].rolling(window=w).std().shift(1)
    return(df)

def ratio_stats(df,metric_type, c, period1=5, period2=365):
    col1 = '{0}_{1}_{2}'.format(c,metric_type,period1)
    col2= '{0}_{1}_{2}'.format(c,metric_type,period2)
    col3= 'ratio_{0}_{1}_{2}_{3}'.format(c,metric_type,period1,period2)
    df[col3]= df.loc[:,col1]/df.loc[:,col2]
    return(df)

def prune_data(df,date_before):
    df= df[df['Date'] > date_before]
    df = df.dropna(axis=0)
    # df.drop(columns= delcols, inplace=True)
    # print(df.columns)
    # newcols = set(df.columns.tolist()).remove('Date')- set(origcols).remove('Date')
    # df = df.loc[:,newcols]
    return(df)

def train_test_split(df,date_before):
    train = df[df['Date'] < date_before]
    test = df[df['Date'] >= date_before]
    return(train,test)

def mae(y,yhat):
    return(sum(abs(y-yhat))/len(y))
def mse(y,yhat):
    return(sum((y-yhat)**2)/len(y))

def date_parts(df,col):
    df['dow']= df.loc[:,col].dt.dayofweek
    df['day_comp']= df.loc[:,col].dt.day
    return(df)
          
def train_predict(train,test,lr,target):
    train_X,train_y= train.loc[:, train.columns !=target],train.loc[:,target]
    test_X,test_y= test.loc[:, test.columns !=target],test.loc[:,target]
    reg= lr.fit(train_X, train_y)
    yhat= lr.predict(test_X)
    return(reg, mae(test_y,yhat), mse(test_y,yhat))
        
def main():
    df,origcols= read_sort('sphist.csv')
    print(origcols)
    windows = [5,30,365]
    wincols = ['Close', 'Volume']
    for c in wincols:
        for w in windows:
            df = rolling_stats(df,c,w)
    win1=5
    win2= 365
    metric_type= ['mean','std']
    for c in wincols:
        for m in metric_type:
            df= ratio_stats(df,m,c,win1,win2)
    df= date_parts(df,'Date')
    df = prune_data(df,'1951-01-02')
    delcols= ['High', 'Low', 'Open', 'Volume', 'Adj Close', 'Date']
    train,test= train_test_split(df,'2013-01-01')
    train.drop(delcols,axis=1,inplace= True)
    test.drop(delcols,axis=1,inplace=True)
    lr= LinearRegression()
    reg,mae,mse= train_predict(train,test,lr,'Close')
    print('____________________________')
    print('MAE: {}. MSE: {}'.format(mae,mse))
    print('Coefficients: \n', reg.coef_)
    print('____________________________')
    
    
if __name__== '__main__':
    main()