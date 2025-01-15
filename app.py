import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components
# Title
# 使用 Markdown 和 HTML 调整字体大小
st.markdown('<h2 style="font-size:20px;">XGBoost Model for Postoperative Thrombosis</h2>', unsafe_allow_html=True)

Age = st.number_input("Age (Year):")
D3Dimer = st.number_input("D3Dimer (μg/mL):")
D5Dimer = st.number_input("D5Dimer (μg/mL):")
FDP = st.number_input("FDP (μg/mL):")
PREDimer = st.number_input("PREDimer (μg/mL):")
PrevWF = st.number_input("PrevWF (ng/mL):")
TT = st.number_input("TT (s):")
lymphocyte = st.number_input("lymphocyte (10^9/L):")
vWFD1 = st.number_input("vWFD1 (ng/mL):")
vWFD3 = st.number_input("vWFD3 (ng/mL):")
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
    input_numerical = np.array([Age, D3Dimer, D5Dimer, Differentiation,FDP, PREDimer, PrevWF, TT, anticoagulation, lymphocyte, vWFD1, vWFD3]).reshape(1, -1)
    feature_names  = ["Age", "D3Dimer", "D5Dimer","Differentiation","FDP","PREDimer","PrevWF","TT","anticoagulation","lymphocyte","vWFD1","vWFD3"]
    input_numericalyuan = pd.DataFrame(input_numerical, columns=feature_names)
    input_numerical = pd.DataFrame(input_numerical, columns=feature_names)

    input_numerical[['D5Dimer','vWFD3','D3Dimer','PrevWF','Age','vWFD1','lymphocyte','TT','FDP','PREDimer']] = scaler.transform(input_numerical[['D5Dimer',
                                                                                                                           'vWFD3','D3Dimer','PrevWF','Age','vWFD1','lymphocyte','TT','FDP','PREDimer']])

        # 使用模型进行预测概率
    prediction_proba = XGB.predict_proba(input_numerical)
    target_class_proba = prediction_proba[:, 1]
    target_class_proba_percent = (target_class_proba * 100).round(2)
    # 在 Streamlit 中显示结果，调整标题样式和内容
    st.markdown("## **Prediction Probabilities (%)**")
    for prob in target_class_proba_percent:
        st.markdown(f"**{prob:.2f}%**")

  
    explainer = shap.TreeExplainer(XGB)
    shap_values = explainer.shap_values(input_numerical)
    
    # SHAP Force Plot
    st.write("### SHAP Value Force Plot")
    shap.initjs()
    force_plot_visualizer = shap.plots.force(
        explainer.expected_value, shap_values, input_numericalyuan)
    # 将 force_plot 保存为一个临时 HTML 文件
    shap.save_html("force_plot.html", force_plot_visualizer)

# 读取 HTML 文件内容
    with open("force_plot.html", "r", encoding="utf-8") as html_file:
        html_content = html_file.read()

# 将 HTML 嵌入到 Streamlit 中
    components.html(html_content, height=400)