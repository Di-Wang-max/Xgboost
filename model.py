import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import streamlit as st
data = pd.read_csv("E:/wangjing/修改后/12模型/VTE.csv", encoding='utf-8')
X = data.drop(columns=['VTE'])
y = data['VTE']
# 假设 X 为特征矩阵，y 为标签
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=123
)
scaler = StandardScaler()
X_train[['D5Dimer','vWFD3','D3Dimer','PrevWF','Age','anticoagulation','vWFD1',
         'Differentiation','lymphocyte','TT','FDP','PREDimer']] = scaler.fit_transform(X_train[['D5Dimer','vWFD3','D3Dimer','PrevWF','Age','anticoagulation','vWFD1',
                              'Differentiation','lymphocyte','TT','FDP','PREDimer']])
X_test[['D5Dimer','vWFD3','D3Dimer','PrevWF','Age','anticoagulation','vWFD1',
        'Differentiation','lymphocyte','TT','FDP','PREDimer']] = scaler.transform(X_test[['D5Dimer','vWFD3','D3Dimer','PrevWF','Age','anticoagulation','vWFD1',
                         'Differentiation','lymphocyte','TT','FDP','PREDimer']])#用训练集的参数转换验证集
#进行排序
X_train = X_train.reindex(sorted(X_train.columns), axis=1)
X_test = X_test.reindex(sorted(X_test.columns), axis=1)
XGB = xgb.XGBClassifier(booster='dart',
                            learning_rate=0.23851708975586638,
                            max_depth=10,
                            min_child_weight= 0.022071308995977257,
                            subsample=0.6574094628645047,
                            colsample_bytree= 0.6010664116285754, 
                            reg_lambda=6.3574413215222005,
                            alpha=0.1273066688719915,
                            n_estimators=376,
                            scale_pos_weight= 7.239614260326147, 
                            max_delta_step= 9,
                   random_state=42)
XGB.fit(X_train, y_train)

import joblib

joblib.dump(XGB, "XGB.pkl")
joblib.dump(scaler, "scaler.pkl")