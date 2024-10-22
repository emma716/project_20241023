import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# streamlit run app/appPc.py
st.title("機器學習分類器")
# 側邊攔
data = st.sidebar.selectbox('### 請選擇資料集 :',['IRIS', 'WINE', 'CANCER'])
clf = st.sidebar.selectbox('### 請選擇分類器 :',['LR', 'SVM', 'KNN', 'RandomForest'])

# 下載資料前先做清空(myData=None)，並取得X,y
def loadData(dd):
    myData = None
    if dd =='IRIS':
        myData = datasets.load_iris()
    elif dd =='WINE':
        myData = datasets.load_wine()
    else:
        myData = datasets.load_breast_cancer()

    X = myData.data
    y = myData.target
    yName = myData.target_names
    return X,y,yName

X, y, yName = loadData(data)   # 進行loadData函式呼叫
st.write('### 資料集結構 :', X.shape)
st.write('### 資料集分類數量 :', len(np.unique(y)))   
st.write('### 資料集分類名稱 :')
for i in yName:
    st.write('#### ', i)
st.write('### 資料集前5筆資料 :', X[:5]) # 也可用st.table(X[:5])

# 定義模型的參數
def model(m):
    p={}
    if m =='LR':
        C = st.sidebar.slider('設定參數C :', 0.01, 10.0,) # C為正則化懲罰項(L1/L2)的倒數
        p['C'] = C
    elif m =='SVM':
        C = st.sidebar.slider('設定參數C :', 0.01, 10.0, key='a') 
        p['C'] = C
    elif m =='KNN':
        K = st.sidebar.slider('設定參數K :', 1, 10) # K為n_neighbors(鄰居數)
        p['K'] = K
    else:
        N = st.sidebar.slider('設定樹的數量 :', 10, 500) # N為n_estimators(幾棵樹)
        D = st.sidebar.slider('設定樹的分析層數 :', 1, 100)
        p['N'] = N
        p['D'] = D
    return p

# 建立模型
ps = model(clf)   # 進行model函式呼叫
def myModel(clf, p):
    new_clf = None
    if clf =='LR':
        new_clf = LogisticRegression(C=p['C'])
    elif clf =='SVM':
        new_clf = SVC(C=p['C'])
    elif clf =='KNN':
        new_clf = KNeighborsClassifier(n_neighbors=p['K'])
    else:
        new_clf = RandomForestClassifier(n_estimators=p['N'], max_depth=p['D'])
    return new_clf

myclf = myModel(clf, ps)   # 進行myModel函式呼叫

# 分割訓練、測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)
# 進行訓練計算+預測
myclf.fit(X_train, y_train)
y_pred = myclf.predict(X_test)

# 進行評分
acc = f'{accuracy_score(y_test, y_pred)*100:.2f}%'
st.write('### 分類準確性 :', acc)

# 降維
pca = PCA(2)
newX = pca.fit_transform(X)

# 繪圖
fig = plt.figure() # 提供放圖的空間(for streamlit)
plt.scatter(newX[:, 0], newX[:, 1], c=y, alpha=0.7)
#plt.show()
st.pyplot(fig)

# pca2 = PCA(n_components=2)
# X_train_pca = pca2.fit_transform(X_train_std)
# X_test_pca = pca2.transform(X_test_std)