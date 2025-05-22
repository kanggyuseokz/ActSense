import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ✅ 절대경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
SCRIPT_DIR = os.path.join(BASE_DIR, 'scripts')

# ✅ 전처리 모듈 경로 추가
sys.path.append(SCRIPT_DIR)
from preprocess import load_csv_file

# ✅ 데이터 불러오기
labels = ['stop', 'walk', 'run', 'stairs']
dfs = [load_csv_file(os.path.join(DATA_DIR, f"{label}.csv"), label) for label in labels]
df = pd.concat(dfs, ignore_index=True)

# ✅ 학습
X = df[['X', 'Y', 'Z']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ 평가
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ✅ 모델 저장
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, 'activity_model.pkl'))
print("✅ 모델 저장 완료:", os.path.join(MODEL_DIR, 'activity_model.pkl'))
