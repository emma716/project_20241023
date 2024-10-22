import numpy as np
import pandas as pd
import streamlit as st

# streamlit run app/test01.py
# 文字顯示
st.title("元件練習操作") #最大標題
st.header("AAAA")       #標題
st.subheader("BBBBB")   #子標題
st.write("Price:",100)  #write類似於python的print
name='Joe'
st.write("Name:",name)  #執行變數
st.write("# 元件練習操作") #後必須空一格再輸入文字，根據#個數來調整字體大小，#最多可到6個，#個數越多字型越小
st.write("## 元件練習操作") 
st.write("### 元件練習操作") 

a = np.array([10,20,30])
st.write(a)
b = pd.DataFrame([[11,22],[33,44]])  #二維
st.write(b)
st.table(b)
st.write(list(range(10)))

# 核取方塊 Checkbox(可勾選選項，單選/複選)
st.write("### 核取方塊Checkbox1----------")
rel = st.checkbox("白天")
if rel:
    #st.write("day")
    st.info("day")
else:
    #st.write("night")
    st.info("night")

st.write("### 核取方塊Checkbox2----------")
checks = st.columns(4) # column為版面劃分，此劃分4等分
with checks[0]:
    c1 = st.checkbox("漢堡")
    if c1:
        st.info("漢堡 checked")
with checks[1]:
    c2 = st.checkbox("薯條")
    if c2:
        st.info("薯條 checked")
with checks[2]:
    c3 = st.checkbox("雞塊")
    if c3:
        st.info("雞塊 checked")
with checks[3]:
    c4 = st.checkbox("蘋果派")
    if c4:
        st.info("蘋果派 checked")

# 選項按鈕 RadioButton(單選)
st.write("### 選項按鈕RadioButton----------")
sex = st.radio("性別:", ("M","F","None"), index=1) #預設選項通常為第一個，若有指定預設哪個選項，需加index
st.info(sex)

sex2 = st.radio("性別:", ("M","F","None"), key="a") #若有重複內容需給key以便識別
st.info(sex2)

st.write("### 選項按鈕RadioButton2 + 數字輸入框")
col1, col2 = st.columns(2)
with col1:
    num1 = st.number_input("請輸入任一數 :")   # 可輸入數字(number)
with col2:
    num2 = st.number_input("請輸入任一整 :", key="b")
ra = st.radio("計算 :", ("＋", "－", "×", "÷"), key="c")
if ra=="＋":
    st.write("{}+{}={}".format(num1,num2,num1+num2))  #{}為格式化第二版的寫法
elif ra=="－":
    st.write("{}-{}={}".format(num1,num2,num1-num2))
elif ra=="×":
    st.write("{}*{}={}".format(num1,num2,num1*num2))
elif ra=="÷":
    st.write("{}/{}={}".format(num1,num2,num1/num2))

# 滑桿 Slider
st.write("### 滑桿Slider----------")
slider = st.slider("請選擇參數 :", 1.0, 20.0, step=0.5)  # 最小值與最大值之間的範圍需大一些
st.info(slider)

slider2 = st.slider("請選擇參數範圍 :", 1.0, 20.0, (11.0,17.0), step=0.01, key="d")
st.info(slider2)

# 下拉選單 SelectBox
st.write("### 下拉選單SelectBox:單選")
s1 = st.selectbox("請選擇城市:", ('台北','台中','台南'), index=1) #預設選項通常為第一個，若有指定預設哪個選項，需加index
"選擇的城市:",s1

# 顯示圖片 Image
st.write("### 顯示圖片")
st.image('app/triangle.png')

# 上傳csv 盡量不做
st.write("### 上傳csv")
file = st.file_uploader("請選擇CSV檔")
if file is not None:
    df = pd.read_csv(file, header=None)
    st.write(df.iloc[:10, :])

# 側邊攔 SideBar
st.write("### 側邊攔SideBar")
name2=st.sidebar.text_input("請輸入名稱:")   # 可輸入文字/數字(text)
st.sidebar.text(name2)   # 標籤
name3=st.sidebar.number_input("請輸入參數:")
st.sidebar.text(name3)

# 按鈕 Button
st.write("### 按鈕Button")
ok = st.button("確定")
if ok:
    st.write("OK")