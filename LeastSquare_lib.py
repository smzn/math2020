'''
Created on Mar 10, 2020

@author: mizuno
'''
import pandas as pd
from numpy.linalg import solve
import matplotlib.pyplot as plt
import numpy as np

class LeastSquare_lib:
    def __init__(self, data, p):
        self.data = data
        self.p = p
        self.df = pd.DataFrame(self.data)

    def getDescribe(self):
        self.df.columns = ['x','y']
        return self.df.describe()

    def getLeft(self):
        left = [[0 for i in range(self.p + 1)] for i in range(self.p + 1)] #要素数pの2次元配列を作成、pはxの次数なので、係数はp+1個ある
        for i in range(len(left)): #係数行列行数
            for j in range(len(left)): #係数行列列数
                for k in self.data[:,0]: #利用データ(x)
                    left[i][j] += k**(i+j)
        return left

    def getRight(self):
        right = [0 for i in range(self.p + 1)]
        for i in range(len(right)): #係数行列行数
            for j in self.df.itertuples(): #利用データ(x,y)
                right[i] += j[1]**i * j[2] #データフレームの要素番号は1から
        return right

    def getSolve(self, left, right):
        value = solve(left, right)
        return value

    def getGraph(self, value):
        plt.scatter(self.data[:,0],self.data[:,1])
        plt.grid()
        x = np.arange(np.min(self.data[:,0]) - 0.5, np.max(self.data[:,0]) + 0.5, 0.1) #xの最小値と最大値の0.5余裕をとる
        y = 0
        for i ,j in enumerate(value): #(i,j) = (要素番号、値)
            y += j * x ** i
        plt.plot(x, y, label = 'p = '+str(self.p ))
        plt.legend() #凡例を追加
        plt.show()

    def getRSS(self, value):
        r2 = 0 #決定係数
        rss = 0 #残差平方和
        sum = 0 #偏差平方和
        r2m = 0 #調整済み決定係数
        #RSS(残差平方和)を求める
        for i, j in zip(self.data[:, 0], self.data[:,1]):
            y = 0
            for k, s in enumerate(value):
                y += s * i ** k
            rss += (j - y)**2
        #偏差平方和を求める
        for i in self.data[:,1]:
            sum += (i - np.mean(self.data[:,1]))**2
        #決定係数を求める
        r2 = 1 - rss / sum
        #調整済み決定係数
        #r2m = 1 - (rss / (len(self.df) - self.p - 1)) / (sum /( len(self.df) - 1)) #len(self.df)は行数, len(self.df.columns)は列数
        r2m = 1 - (rss / sum) * ((len(self.df) - 1) / (len(self.df) - (len(self.df.columns)-1) -1))
        # r2m = 1 - (rss / n - k -1) / (sum / n - 1) n:データ数、k:説明変数の数(len(self.df.column) -1)列数だとyも入る
        return rss, r2, r2m

