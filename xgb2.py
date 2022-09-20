#%%
import xgboost as xgb
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#%%
wd='/Users/shruti/Desktop/机器学习论文/训练数据和测试数据'
os.chdir(wd)
mydata = pd.read_csv('./Mydata_train_1034_13-17.csv', index_col=False, encoding='utf-8')
data = xgb.DMatrix(mydata)
print(data)
#%%
x = data.drop(columns='ma')  # 特征
y = data['ma']  # 标签
#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.8)  # 划分数据集
print('训练集和测试集 shape', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#%%
dtrain = xgb.DMatrix(x_train, label = y_train)
dtest = xgb.DMatrix(x_test, label = y_test)

#%%
param = {'max_depth':2, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
train_preds = bst.predict(dtrain)
#%%
train_predictions = [round(value) for value in train_preds] #进行四舍五入的操作--变成0.1(算是设定阈值的符号函数)
train_accuracy = accuracy_score(y_train, train_predictions) #使用sklearn进行比较正确率
print ("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
#%%
from xgboost import plot_importance #显示特征重要性
plot_importance(bst)#打印重要程度结果。
pyplot.show()