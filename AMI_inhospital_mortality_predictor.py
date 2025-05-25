#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

current_dir = os.path.dirname(__file__)  # 获取当前脚本所在目录
file_path = os.path.join(current_dir, "LightGBM.pkl")
with open(file_path, "r") as f:
    model = joblib.load(file_path)

# 定义特征名称
feature_names = ['Age', 'TIT', 'Diabetes', 'Hypertension', 'Hyperlipidemia',
       'Smoking_history', 'BMI', 'SBP', 'DBP', 'HR', 'CREA', 'UA', 'RBG',
       'LDLC', 'eGFR', 'CRP', 'HCT', 'NEUT', 'LYMPH', 'NLR', 'HGB', 'PLT',
       'MYO', 'CKMB', 'TNI', 'NTproBNP', 'HbA1c', 'LAD', 'LVEF', 'LVEDV',
       'LVESV', 'CS']

# Streamlit 用户界面
st.title("AMI In-hospital Mortality Predictor")

# 用户输入
Age = st.number_input("Age:", min_value=18, max_value=100, value=50)
Gender = st.selectbox("Gender (0=Female, 1=Male):", options=[0, 1],
                   format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')
Mass = st.number_input("Mass,Kg:", min_value=20.0, max_value=300.0, value=70.0)
TIT = st.number_input("Total Ischeamia Time,hour:", min_value=1, max_value=2000, value=10)
Diabetes = st.selectbox("Diabetes (0=No, 1=Yes):", options=[0, 1],
                   format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Hypertension = st.selectbox("Hypertension (0=Nbody mass indexo, 1=Yes):", options=[0, 1],
                   format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Hyperlipidemia = st.selectbox("Hyperlipidemia (0=No, 1=Yes):", options=[0, 1],
                   format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Smoking_history = st.selectbox("Smoking history  (0=No, 1=Yes):", options=[0, 1],
                   format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
BMI = st.number_input("body mass index,Kg/m2:", min_value=18, max_value=500, value=24)
SBP = st.number_input("systolic blood pressure,mmHg:", min_value=0, max_value=300, value=120)
DBP = st.number_input("diastolic blood pressure,mmHg:", min_value=0, max_value=200, value=70)
HR = st.number_input("heart rate,beats/minute:", min_value=0, max_value=200, value=70)
CREA = st.number_input("creatinine,umol/L:", min_value=1, max_value=2000, value=72)
UA = st.number_input(" uric acid,umol/L:", min_value=10, max_value=2000, value=345)
RBG = st.number_input("random blood glucose,mmol/L:", min_value=0.0, max_value=35.0, value=6.8)
LDLC = st.number_input("LDLC,mmol/L:", min_value=0.0, max_value=20.0, value=3.0)
if Gender == 1:
    eGFR = (140 - float(Age)) * float(Mass) * 1.23 / float(CREA)
else:
    eGFR = (140 - float(Age)) * float(Mass) * 1.03 / float(CREA)
CRP = st.number_input("CRP,mg/L:", min_value=0.0, max_value=500.0, value=10.0)
HCT = st.number_input("HCT,%:", min_value=0.0, max_value=100.0, value=46.0)
NEUT = st.number_input("NEUT,10e9/L:", min_value=0.0, max_value=30.0, value=7.4)
LYMPH = st.number_input("LYMPH,10e9/L:", min_value=0.0, max_value=10.0, value=1.4)
NLR = NEUT/LYMPH
HGB = st.number_input("HGB,g/L:", min_value=0.0, max_value=250.0, value=150.0)
PLT = st.number_input("PLT ,10e9/L:", min_value=0.0, max_value=1000.0, value=180.0)
MYO = st.number_input("MYO,ng/mL:", min_value=0.0, max_value=900.0, value=280.0)
CKMB = st.number_input("CKMB,ng/mL:", min_value=0.0, max_value=900.0, value=62.0)
TNI = st.number_input("TNI,ng/mL:", min_value=0.0, max_value=100.0, value=2.0)
NTproBNP = st.number_input("NTproBNP,pg/mL:", min_value=0.0, max_value=35000.0, value=560.0)
HbA1c = st.number_input("HbA1c,%:", min_value=3.0, max_value=20.0, value=5.8)
LAD = st.number_input("LAD,cm:", min_value=1.0, max_value=10.0, value=3.0)
LVEF = st.number_input("LVEF,%:", min_value=5, max_value=80, value=50)
LVEDV = st.number_input("LVEDV,ml:", min_value=30.0, max_value=500.0, value=125.0)
LVESV = st.number_input("LVESV,ml:", min_value=10.0, max_value=200.0, value=57.0)
CS = st.selectbox("Cardiogenic Shock (0=No, 1=Yes):", options=[0, 1],
                   format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
# 处理输入并进行预测
feature_values = [
    Age, TIT, Diabetes, Hypertension, Hyperlipidemia, Smoking_history,
    BMI, SBP, DBP, HR, CREA, UA, RBG, LDLC, eGFR, CRP, HCT, NEUT, LYMPH, NLR,
    HGB, PLT, MYO, CKMB, TNI, NTproBNP, HbA1c, LAD, LVEF, LVEDV, LVESV, CS
]

# 将feature_values转换为NumPy数组，确保所有数据都是数值类型
features = np.array([feature_values], dtype=np.float32)

if st.button("Predict"):
    # 预测类别和概率
    predicted_proba = model.predict_proba(features)[0]
    # 显示预测结果
    st.write(f"**Prediction Probabilities:** {predicted_proba[1]:.2f}")
    # 根据预测结果生成建议
    probability = predicted_proba[1] * 100
    advice = (
        f"According to our model, there is a potential risk of in-hospital mortality for the patient."
        f"The model estimates that the likelihood of this outcome is approximately {probability:.1f}%. "
    )
    st.write(advice)
    # 计算SHAP值并显示力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names),
                    matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")