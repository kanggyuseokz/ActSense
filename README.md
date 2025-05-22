# ActSense

스마트폰 센서 데이터를 기반으로 사용자의 활동 상태 (정지, 걷기, 뛰기, 계단 오르기)를 분류하는 머신러닝 + 웹 애플리케이션

## 구성

- RandomForestClassifier 모델 사용
- CSV 데이터 업로드 기반 예측 (Streamlit 웹 UI)

## 실행 방법

1. `pip install -r requirements.txt`
2. 모델 학습: `python scripts/train_model.py`
3. 웹 실행: `streamlit run app/web_app.py`

---
## 디렉토리 구조
```
ActSense/
│
├── data/ # ▶ 원시 센서 데이터 (CSV 파일)
│ ├── stop.csv # → 정지 상태 센서 데이터
│ ├── walk.csv # → 걷기 상태 센서 데이터
│ ├── run.csv # → 뛰기 상태 센서 데이터
│ ├── stairs.csv # → 계단 오르기 상태 센서 데이터
│
├── model/ # ▶ 학습된 모델 파일 저장 폴더
│ └── activity_model.pkl # → 학습 완료된 Random Forest 모델
│
├── scripts/ # ▶ 학습 및 전처리 관련 코드
│ ├── train_model.py # → 데이터 로딩, 학습, 평가, 모델 저장 수행
│ └── preprocess.py # → 데이터 로딩 및 전처리 함수 정의
│
├── app/ # ▶ 웹 앱 실행 코드
│ └── web_app.py # → Streamlit 기반 CSV 업로드 및 예측 UI
│
├── requirements.txt # ▶ 프로젝트 의존성 목록
└── README.md # ▶ 현재 문서
```