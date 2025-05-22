import pandas as pd
from scripts.preprocess import load_csv_file
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# 데이터 불러오기
data_dir = '../data'
labels = ['stop', 'walk', 'run', 'stairs']
dfs = [load_csv_file(f"{data_dir}/{label}.csv", label) for label in labels]
df = pd.concat(dfs, ignore_index=True)

# 특성, 타깃 분리
X = df[['X', 'Y', 'Z']]
y = df['label']

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 평가 출력
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 모델 저장
os.makedirs('../model', exist_ok=True)
joblib.dump(model, '../model/activity_model.pkl')