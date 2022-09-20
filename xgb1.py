#%%
import os
import pandas as pd
import xgboost as xgb
import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import pearsonr
#%%
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#%%
wd='/Users/shruti/Desktop/机器学习论文/训练数据和测试数据'
os.chdir(wd)
#%%%
mydata = pd.read_csv('./Mydata_train_1034_13-17.csv', index_col=False, encoding='utf-8')
print(mydata.head(5))
#%%
x = mydata.drop(columns='ma')  # 特征
y = mydata['ma']  #标签
#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.8)  # 划分数据集
print('训练集和测试集 shape', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#%%
model = xgb.XGBRegressor(max_depth=6,learning_rate=0.05,n_estimators=100,randam_state=42)
#%%
model.fit(x_train, y_train)  # 在训练集上训练模型
print(model)  # 输出模型XGBClassifier()
#%%
# 在测试集上测试模型
predicted = model.predict(x_test)
#%%
mypredict=pd.DataFrame(predicted,columns=['Predicted'])
mypredict.reset_index(drop=True, inplace=True)
myobserve=pd.DataFrame(y_test)
myobserve.rename(columns={'ma':'Observed'},inplace=True)
myobserve.reset_index(drop=True, inplace=True)

# In[]
draw = pd.concat([myobserve, mypredict],axis=1)
#%%
metrics_test_real = pd.DataFrame()
#metrics_test_real.index.name = sspfile[0:-4]
metrics_test_real['MAE'] = [mean_absolute_error(draw['Observed'], draw['Predicted'])]
metrics_test_real['MSE'] = [mean_squared_error(draw['Observed'], draw['Predicted'])]
metrics_test_real['RMSE'] = [np.sqrt(mean_squared_error(draw['Observed'], draw['Predicted']))]
metrics_test_real['r2_score'] = [r2_score(draw['Observed'], draw['Predicted'])]
metrics_test_real['r'] = pearsonr(draw['Observed'], draw['Predicted'])[0]
metrics_test_real['p-value'] = pearsonr(draw['Observed'], draw['Predicted'])[1]
print(metrics_test_real)
#%%输出结果
draw.to_csv('D:/machine/data/RFresult/python_ssp2020predict/' + sspfile, encoding="utf-8", header=True, index=False)

