import pandas as pd

def load_csv_file(filepath, label=None):
    """
    센서 CSV 파일을 읽어 X, Y, Z 컬럼만 남기고 정제합니다.
    
    Parameters:
        filepath (str): CSV 파일 경로
        label (str, optional): 레이블 이름 (예: 'stop', 'walk')
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    # CSV 읽기 (헤더 없다고 가정)
    df = pd.read_csv(filepath, header=None)

    # 컬럼 이름 수동 지정
    df.columns = ['Time', 'X', 'Y', 'Z', 'Absolute']

    # 필요한 컬럼만 추출
    df = df[['X', 'Y', 'Z']]

    # 데이터 타입 변환 및 결측치 제거
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    # 레이블 추가
    if label:
        df['label'] = label

    return df
