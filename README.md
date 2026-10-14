# ANN Prediction Analyzer

ANN Prediction Analyzer는 사용자가 수치형 데이터를 입력하고 **Target / Features / Hidden neuron / Activation을 직접 설정한 뒤,
Dense 기반 ANN을 학습·평가·예측할 수 있는 Streamlit 웹 프로그램**입니다.

## 프로그램이 하는 일

1. 예제 데이터 또는 CSV/Excel 입력
2. Target / Features 선택
3. Train / Validation / Test 분리
4. Train 기준 Standardization
5. Dense ANN 생성
6. MSE + Backpropagation + Adam으로 학습
7. EarlyStopping으로 가장 좋은 Weight 선택
8. Test 데이터에서 R² / RMSE / MAE 평가
9. Actual vs Predicted / Residual / Learning Curve 표시
10. Weight / Bias 확인
11. 새로운 입력값 예측

내장 예제 데이터는 프로그램 기능을 바로 실습하기 위한 보조 데이터이며,
프로그램 자체는 다른 수치형 회귀 데이터에도 사용할 수 있습니다.


Dense 기반 인공신경망의 **Weight 학습 원리 → 표준화 → 구조 → 학습 → 평가 → 예측**을 단계별로 확인할 수 있는 Streamlit 프로그램입니다.

## 기본 실습 구조

`Input(3) → Dense Hidden(3) → Activation → Dense Output(1)`

입력 3개와 Hidden neuron 3개를 사용하면 Input→Hidden에는 9개의 Weight,
Hidden→Output에는 3개의 Weight가 존재합니다.

## 핵심 계산

Dense:
`z = XW + b`

Activation:
`a = φ(z)`

Loss:
`MSE = (1/n) Σ(y-ŷ)²`

Learning:
`Forward → Loss → Backpropagation → Adam → Weight/Bias Update`

## 표준화

표준화는 ANN Layer가 아니라 ANN 입력 전 전처리입니다.

`x' = (x - Train Mean) / Train Std`

Train 데이터에만 fit하고 Validation/Test/새 입력값에는 같은 기준을 적용합니다.

## 실습용 예제 데이터

scikit-learn Diabetes 데이터를 `scaled=False`로 불러와 원래 단위에 가까운 값으로 표시합니다.

기본 Feature:
- BMI
- Blood_Pressure
- Blood_Sugar

Target:
- Disease_Progression

프로그램 내부에서 Train 데이터를 기준으로 별도 표준화를 수행합니다.

## 공개 구현 참고자료

- Keras Dense: https://keras.io/api/layers/core_layers/dense/
- Keras Adam: https://keras.io/api/optimizers/adam/
- Keras EarlyStopping: https://keras.io/api/callbacks/early_stopping/
- Keras ReduceLROnPlateau: https://keras.io/api/callbacks/reduce_lr_on_plateau/

## 실행

```bash
python -m pip install -r requirements.txt
python -m streamlit run ANN_Prediction_Analyzer.py
```


## 상세 설명 버전 추가

Streamlit 화면에는 초보자용 ANN 개념 설명, 용어 사전, R²/RMSE/MAE 해석, Learning Curve 해석, 표준화 설명, Hidden neuron/Feature 선택 가이드를 추가했습니다.

특히 R²는 정확도/확률이 아니라 Test 데이터의 변동을 설명하는 정도이며, 결과는 R² 하나가 아니라 RMSE, MAE, Validation Loss, Residual과 함께 해석하도록 구성했습니다.
