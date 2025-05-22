import streamlit as st
import pandas as pd
import joblib
from scripts.preprocess import load_csv_file

# 모델 로드
model = joblib.load('../model/activity_model.pkl')

st.title("🏃 활동 분류기")
st.write("센서 CSV 데이터를 업로드하면 활동을 예측합니다.")

uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")

if uploaded_file:
    try:
        # 업로드된 파일 전처리
        df = pd.read_csv(uploaded_file, header=None)
        df.columns = ['Time', 'X', 'Y', 'Z', 'Absolute']
        df = df[['X', 'Y', 'Z']]
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        # 예측 수행
        preds = model.predict(df)
        pred_mode = pd.Series(preds).mode()[0]

        st.success(f"예측된 활동 상태: **{pred_mode.upper()}**")
        st.bar_chart(pd.Series(preds).value_counts())

    except Exception as e:
        st.error(f"오류 발생: {e}")