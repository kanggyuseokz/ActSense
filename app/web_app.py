import os
import sys
import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import importlib.util

# === 경로 설정 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCRIPT_DIR = os.path.join(BASE_DIR, 'scripts')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'activity_model.pkl')
PREPROCESS_PATH = os.path.join(BASE_DIR, 'scripts', 'preprocess.py')
# === 전처리 모듈 강제 import
spec = importlib.util.spec_from_file_location("preprocess", PREPROCESS_PATH)
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)
load_csv_file = preprocess.load_csv_file

# === Streamlit UI 시작 ===
st.title("🏃 활동 분류기")
st.markdown("CSV 센서 데이터를 업로드하면 활동 상태(정지, 걷기, 뛰기, 계단)를 예측합니다.")

# 파일 업로드
uploaded_file = st.file_uploader("📤 CSV 파일을 업로드하세요", type="csv")

# 모델 로드
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ 모델 로드 실패: {e}")
    st.stop()

# 업로드된 파일이 있으면 예측 수행
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        df.columns = ['Time', 'X', 'Y', 'Z', 'Absolute']
        df = df[['X', 'Y', 'Z']]
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        preds = model.predict(df)
        pred_mode = pd.Series(preds).mode()[0]

        st.success(f"📌 예측된 대표 활동: **{pred_mode.upper()}**")
        st.bar_chart(pd.Series(preds).value_counts())

    except Exception as e:
        st.error(f"❌ 예측 중 오류: {e}")