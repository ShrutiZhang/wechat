##导入库
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import r2_score#R2
from math import sqrt
path='D:/machine/data/训练数据和测试数据/' #文件路径
print(path)
os.chdir(path)  #修改工作路径到path下
#%%
#训练数据准备
trainset=pd.read_csv(path+'Mydata_train_prd_13-17.csv',index_col=None,header=0).dropna()
#print(trainset)
###训练数据准备
train_Y=trainset.loc[:,['ma']]
train_X=trainset.drop(columns=['ma'])
print(train_X)
#%%建立RF模型
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators = 500,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                           random_state = 90)
rf.fit(train_X, train_Y.values.ravel())
#%%
###特征重要性评估
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = train_X.columns[:]
print('Importance')
for f in range(train_X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 60, feat_labels[indices[f]],importances[indices[f]]))
#%%预测数据集 Predict_X
datapath = path
# 待搜索的名称
filename = "2020_SSP"
# 定义保存结果的数组
result = []
    ##自定义findfile 函数,s输出结果保存在result的list中
    def findfiles(files_path, files_list):
        files = os.listdir(files_path)
        for i in files:
            i_path = os.path.join(files_path, i)
            if os.path.isfile(i_path) and filename in i:
                result.append(i)

    if __name__ == '__main__':
        findfiles(datapath, result)
print(result)
#%%读取2020年的观测数据
test_Y = pd.read_csv('D:/machine/data/训练数据和测试数据/2020_ozonetest.csv', index_col=None, header=0)
print(test_Y)
#%%
from scipy.stats import pearsonr
for i in range(len(result)):
    sspfile = result[i]
    predict_X = pd.read_csv(datapath + sspfile, index_col=None, header=0).dropna()[train_X.columns]
    predict_X2=predict_X[-predict_X.month.isin([202012])]
    predicted_ambient = rf.predict(predict_X2)
    predicted_ambient = pd.DataFrame(predicted_ambient, columns=['predict'])
    draw = pd.concat([pd.DataFrame(test_Y), predicted_ambient], axis=1)
    metrics_test_real = pd.DataFrame()
    #metrics_test_real.index.name = sspfile[0:-4]
    metrics_test_real['MAE'] = [mean_absolute_error(draw['ma'], draw['predict'])]
    metrics_test_real['MSE'] = [mean_squared_error(draw['ma'], draw['predict'])]
    metrics_test_real['RMSE'] = [np.sqrt(mean_squared_error(draw['ma'], draw['predict']))]
    metrics_test_real['r2_score'] = [r2_score(draw['ma'], draw['predict'])]
    metrics_test_real['r'] = pearsonr(draw['ma'],draw['predict'])[0]
    metrics_test_real['p-value'] = pearsonr(draw['ma'],draw['predict'])[1]
    print(result[i])
    print(metrics_test_real)
    draw.to_csv('D:/machine/data/RFresult/python_ssp2020predict/' + sspfile, encoding="utf-8", header=True,index=False)
    #metrics_test_real.to_csv('D:/machine/data/RFresult/python_ssp2020predict/index' + sspfile, encoding="utf-8", header=True,index=True)
print(draw.head(5))
#%%