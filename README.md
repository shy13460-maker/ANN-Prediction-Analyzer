# 🧠 ANN Prediction Analyzer

수치형 데이터를 이용해 **Dense 기반 인공신경망 회귀 모델을 구성·학습·평가·예측**하는 Streamlit 웹 프로그램입니다.

이 프로젝트의 중심은 특정 데이터셋이 아니라 **ANN의 구조와 학습 알고리즘**입니다.

## 핵심 알고리즘

`Data`
→ `Preprocessing`
→ `Train / Validation / Test`
→ `Scaling`
→ `Input Layer`
→ `Dense Hidden Layer`
→ `Activation Function`
→ `Dense Hidden Layer`
→ `Activation Function`
→ `Linear Output`
→ `MSE Loss`
→ `Backpropagation`
→ `Adam`
→ `Evaluation`
→ `Prediction`

## Dense 계산

Dense layer:

`z = XW + b`

Activation:

`a = φ(z)`

프로그램에서는 계산 흐름이 명확히 보이도록 Dense와 Activation을 별도 Layer로 구성했습니다.

## 주요 기능

- CSV / XLSX / XLS 업로드
- 실습용 예제 데이터 내장
- Feature / Target 선택
- Train / Validation / Test 분리
- StandardScaler / RobustScaler
- Hidden Layer 직접 설정
- ReLU / tanh / sigmoid / ELU
- Dropout / L2
- Adam Optimizer
- EarlyStopping
- ReduceLROnPlateau
- R² / RMSE / MAE
- Actual vs Predicted
- Residual Plot
- Learning Curve
- Linear Regression baseline
- Permutation Feature Importance
- Dense Weight / Bias 확인
- 새로운 데이터 예측
- Train 범위 밖 입력 경고
- Excel / JSON 저장

## 공개된 구현 참고자료

- Keras Dense  
  https://keras.io/api/layers/core_layers/dense/
- Keras Adam  
  https://keras.io/api/optimizers/adam/
- Keras EarlyStopping  
  https://keras.io/api/callbacks/early_stopping/
- Keras ReduceLROnPlateau  
  https://keras.io/api/callbacks/reduce_lr_on_plateau/

## 실습용 예제 데이터

프로그램 사용법을 바로 확인할 수 있도록 scikit-learn에 포함된 공개 회귀 예제 데이터를 사용합니다.

예제 데이터는 어디까지나 **프로그램 실습용**이며,
사용자는 자신의 CSV/Excel 수치형 데이터를 업로드해 동일한 ANN 분석을 수행할 수 있습니다.

## 실행

```bash
python -m pip install -r requirements.txt
python -m streamlit run ANN_Prediction_Analyzer.py
```

## Streamlit 배포

- Branch: `main`
- Main file: `ANN_Prediction_Analyzer.py`
- Python: 3.12 권장

