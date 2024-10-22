import streamlit as st
import joblib

#streamlit run app/iris_pred_2.py
st.title("鳶尾花品種預測")

# 載入模型
svm = joblib.load("app/svm_model.joblib")
knn = joblib.load("app/knn_model.joblib")
lr = joblib.load("app/lr_model.joblib")

# 選擇模型
clf = st.selectbox("請選擇模型 : ", ("SVM", "KNN", "LogisticRegression"))
if clf=="SVM":
    model = svm
elif clf=="KNN":
    model = knn
else:
    model = lr

# 設定元件
s1 = st.slider("花萼長度 : ", 4.0, 8.0, 6.0)   # (最小值, 最大值, 初始值)
s2 = st.slider("花萼寬度 : ", 2.0, 5.0, 3.5)
s3 = st.slider("花瓣長度 : ", 1.0, 7.0, 4.0)
s4 = st.slider("花瓣寬度 : ", 0.1, 3.0, 1.5)

# 使用模型進行預測
labels = ['setosa', 'versicolor', 'virginica']

if st.button("進行預測"):
    X = [[s1,s2,s3,s4]]   # 特徵必須為二維型態
    y = model.predict(X)
    st.write("### 分類標籤 : ", y[0])
    st.write("### 預測結果 : ", labels[y[0]])