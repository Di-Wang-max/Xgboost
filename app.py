import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components
# Title
# 使用 Markdown 和 HTML 调整字体大小
st.markdown('<h2 style="font-size:20px;">XGBoost Model for Postoperative Thrombosis</h2>', unsafe_allow_html=True)

age = st.number_input("Age:", min_value=1, max_value=120, value=50)
D3Dimer = st.number_input("D3Dimer:")
D5Dimer = st.number_input("D5Dimer:")
FDP = st.number_input("FDP:")
PREDimer = st.number_input("PREDimer:")
PrevWF = st.number_input("PrevWF:")
TT = st.number_input("TT:")
lymphocyte = st.number_input("lymphocyte:")
vWFD1 = st.number_input("vWFD1:")
vWFD3 = st.number_input("vWFD3:")
anticoagulation = st.selectbox('anticoagulation', ['No', 'Yes'])
Differentiation = st.selectbox('Differentiation', ['low',"medium",'high'])
Differentiationmap = {'low': 0, 'medium': 1, 'high': 2}
Differentiation = Differentiationmap[Differentiation]
anticoagulation = 1 if anticoagulation == 'Yes' else 0

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    XGB = joblib.load("XGB.pkl")
    scaler = joblib.load("scaler.pkl")
    
    # Store inputs into dataframe
    input_categorical = np.array([Differentiation,anticoagulation]).reshape(1, -1)
    input_numerical = np.array([age, D3Dimer, D5Dimer,FDP,PREDimer,PrevWF,TT,lymphocyte,vWFD1,vWFD3]).reshape(1, -1)
    input_numerical_scaled = scaler.transform(input_numerical)
     # 将数值和分类变量按顺序整合
    combined_input = np.hstack((input_numerical_scaled[:, 0:3], input_categorical[:, 0:1], 
                                input_numerical_scaled[:, 3:7], input_categorical[:, 1:],
                                input_numerical_scaled[:, 7:]))
    features = np.hstack((input_numerical[:, 0:3], input_categorical[:, 0:1], 
                                input_numerical[:, 3:7], input_categorical[:, 1:],
                                input_numerical[:, 7:]))
    feature_names  = ["age", "D3Dimer", "D5Dimer","Differentiation","FDP","PREDimer","PrevWF","TT","anticoagulation","lymphocyte","vWFD1","vWFD3"]
    features_named = pd.DataFrame(features, columns=feature_names)
    # 使用模型进行预测概率
    prediction_proba = XGB.predict_proba(combined_input)
    target_class_proba = prediction_proba[:, 1]
    target_class_proba_percent = (target_class_proba * 100).round(2)
    # 在 Streamlit 中显示结果，调整标题样式和内容
    st.markdown("## **Prediction Probabilities (%)**")
    for prob in target_class_proba_percent:
        st.markdown(f"**{prob:.2f}%**")

   
    #feature_df = pd.DataFrame(features, columns=feature_names)
    explainer = shap.TreeExplainer(XGB)
    shap_values = explainer.shap_values(combined_input)
    
    # SHAP Force Plot
    st.write("### SHAP Value Force Plot")
    shap.initjs()
    force_plot_visualizer = shap.plots.force(
        explainer.expected_value, shap_values, features_named)
    # 将 force_plot 保存为一个临时 HTML 文件
    shap.save_html("force_plot.html", force_plot_visualizer)

# 读取 HTML 文件内容
    with open("force_plot.html", "r", encoding="utf-8") as html_file:
        html_content = html_file.read()

# 将 HTML 嵌入到 Streamlit 中
    components.html(html_content, height=400)