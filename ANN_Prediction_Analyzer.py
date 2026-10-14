from __future__ import annotations

import io
import json
import math
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="ANN Prediction Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {
    max-width: 1480px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}
[data-testid="stMetric"] {
    border: 1px solid rgba(49, 51, 63, 0.14);
    border-radius: 14px;
    padding: 14px 16px;
}
.step-card {
    border: 1px solid rgba(37, 99, 235, 0.22);
    border-radius: 14px;
    padding: 14px 16px;
    background: rgba(239, 246, 255, 0.70);
    min-height: 88px;
}
.flow-card {
    border: 1px solid rgba(49, 51, 63, 0.14);
    border-radius: 14px;
    padding: 16px 18px;
    margin: 8px 0 18px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧠 ANN Prediction Analyzer")
st.caption(
    "Dense 기반 인공신경망의 데이터 입력 → 학습 → 평가 → 예측 과정을 "
    "직접 확인할 수 있는 범용 회귀 실습 프로그램"
)

st.markdown(
    """
<div class="flow-card">
<b>전체 ANN 파이프라인</b><br>
Data → Train / Validation / Test → Standardization →
Input Layer → Dense Hidden Layer → Activation Function →
Output Layer → Loss → Backpropagation → Adam → Evaluation → Prediction
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
for col, n, title, desc in [
    (c1, "①", "데이터", "예제 또는 업로드"),
    (c2, "②", "변수 선택", "Feature / Target"),
    (c3, "③", "표준화", "Train 기준"),
    (c4, "④", "ANN 구조", "Dense / Activation"),
    (c5, "⑤", "학습·평가", "Loss / Backprop"),
    (c6, "⑥", "예측", "새 입력값"),
]:
    with col:
        st.markdown(
            f"""
<div class="step-card">
<b>{n} {title}</b><br>
<small>{desc}</small>
</div>
""",
            unsafe_allow_html=True,
        )

st.info(
    "처음 실습할 때는 기본값을 그대로 사용하면 됩니다. "
    "고급 설정은 왼쪽 사이드바에서 필요할 때만 펼치면 됩니다."
)


st.header("📌 이 프로그램은 무엇을 하는가?")

st.markdown(
    """
**ANN Prediction Analyzer**는 특정 데이터 한 종류만 분석하는 프로그램이 아니라,
사용자가 수치형 데이터를 넣고 **ANN 회귀 모델을 직접 구성·학습·평가·예측**할 수 있도록 만든 실습형 웹 프로그램입니다.

프로그램에서 사용자는 다음 작업을 수행합니다.

1. **데이터 준비**  
   내장 예제 데이터를 사용하거나 CSV / Excel 파일을 직접 업로드합니다.

2. **예측 문제 정의**  
   무엇을 예측할지 `Target`으로 선택하고, 예측에 사용할 입력정보를 `Features`로 선택합니다.

3. **ANN 구조 설정**  
   Hidden neuron 수와 Activation Function을 선택하여 신경망의 크기와 계산 구조를 정합니다.

4. **자동 전처리 및 학습**  
   프로그램이 데이터를 Train / Validation / Test로 나누고, Train 기준 표준화를 수행한 뒤 ANN을 학습합니다.

5. **모델 평가**  
   학습에 사용하지 않은 Test 데이터에서 R², RMSE, MAE와 그래프를 이용해 예측성능을 확인합니다.

6. **모델 내부 확인**  
   실제 학습된 Weight / Bias와 표준화 전후 데이터를 확인할 수 있습니다.

7. **새로운 데이터 예측**  
   학습이 끝난 모델에 새로운 Feature 값을 넣어 Target을 예측합니다.

**즉, 이 프로그램의 핵심은**  
`데이터 입력 → ANN 구성 → 학습 → 성능평가 → 내부 파라미터 확인 → 새 데이터 예측`
을 하나의 웹 화면에서 직접 실습하는 것입니다.
    """
)

with st.expander("🗺️ 프로그램 화면은 이렇게 구성되어 있습니다", expanded=True):
    screen_table = pd.DataFrame(
        [
            ["왼쪽 사이드바", "데이터 종류, Hidden neuron, Activation, Learning rate 등 ANN 설정"],
            ["1. 데이터 확인", "Target / Features 선택, 실제 분석 데이터 확인 및 CSV 다운로드"],
            ["2. 표준화", "ANN에 넣기 전에 데이터 크기를 맞추는 이유와 처리 방법 설명"],
            ["3. ANN 구조 확인", "현재 선택한 Feature 수, Hidden neuron 수, Weight 개수를 표로 확인"],
            ["4. ANN 학습 및 평가", "버튼을 눌러 데이터 분리부터 학습·Test 평가까지 실행"],
            ["5. 결과 분석", "R², RMSE, MAE, Learning Curve, 표준화, Baseline, Weight/Bias 확인"],
            ["새 데이터 예측 탭", "새 Feature 값을 넣어 학습된 ANN의 예측값 확인"],
        ],
        columns=["화면", "무엇을 하는 곳인가"],
    )
    st.dataframe(screen_table, use_container_width=True, hide_index=True)

with st.expander("▶️ 'ANN 학습 및 평가 실행' 버튼을 누르면 실제로 무엇을 하나?", expanded=True):
    st.markdown(
        """
버튼 한 번으로 다음 작업이 순서대로 실행됩니다.

**① 선택한 Feature와 Target을 배열로 변환**  
→ **② Train / Validation / Test로 분리**  
→ **③ Train 데이터만 이용해 StandardScaler 학습**  
→ **④ Train / Validation / Test를 같은 기준으로 표준화**  
→ **⑤ Input → Dense → Activation → Output 구조의 ANN 생성**  
→ **⑥ Forward Propagation으로 예측값 계산**  
→ **⑦ 실제값과 비교해 MSE Loss 계산**  
→ **⑧ Backpropagation으로 각 Weight/Bias의 Gradient 계산**  
→ **⑨ Adam Optimizer가 Weight/Bias 수정**  
→ **⑩ 여러 Epoch 반복**  
→ **⑪ Validation Loss를 감시하고 EarlyStopping 적용**  
→ **⑫ 가장 좋았던 Weight 복원**  
→ **⑬ Test 데이터로 R² / RMSE / MAE 계산**  
→ **⑭ 그래프와 표로 결과 표시**

즉 이 버튼은 단순히 '예측'만 하는 버튼이 아니라,
**전처리 → 모델 생성 → 학습 → 검증 → 최종 평가** 전체 과정을 실행하는 버튼입니다.
        """
    )

with st.expander("📊 결과 탭은 각각 무엇을 보여주나?", expanded=True):
    result_table = pd.DataFrame(
        [
            ["예측 성능", "실제값과 예측값이 얼마나 가까운지 확인. Actual vs Predicted / Residual 제공"],
            ["학습 과정", "Epoch에 따라 Train Loss / Validation Loss가 어떻게 변했는지 확인"],
            ["표준화 확인", "원래 입력값과 ANN에 실제로 들어간 표준화 값 비교"],
            ["선형모델 비교", "ANN이 단순 Linear Regression보다 실제로 나은지 비교"],
            ["Weight / Bias", "ANN이 학습을 통해 찾은 실제 파라미터 확인"],
            ["새 데이터 예측", "학습에 없던 새로운 Feature 값을 입력해 Target 예측"],
        ],
        columns=["탭", "확인하는 내용"],
    )
    st.dataframe(result_table, use_container_width=True, hide_index=True)

st.info(
    "내장 Diabetes 데이터는 이 프로그램의 '주제'가 아니라 ANN 학습·평가 기능을 바로 확인하기 위한 실습용 예제입니다. "
    "CSV / Excel을 업로드하면 같은 프로그램을 다른 수치형 회귀 데이터에도 사용할 수 있습니다."
)


with st.expander("📘 처음 보는 사람을 위한 ANN 핵심 개념", expanded=True):
    st.markdown(
        """
### ANN이란?

인공신경망(Artificial Neural Network, ANN)은 **입력값을 이용해 예측값을 만들고, 실제 정답과의 차이를 줄이도록 Weight와 Bias를 반복해서 수정하는 모델**입니다.

가장 단순하게 보면 다음 과정입니다.

**입력값 → 예측 → 정답과 비교 → 오차 계산 → Weight/Bias 수정 → 다시 예측**

예를 들어 `100 km → 62.137 miles` 관계를 모른다고 가정하면, 처음 Weight가 `0.5`일 때 `100×0.5=50`을 예측합니다. 정답 62.137과 차이가 크므로 Weight를 수정합니다. Weight가 0.6이면 60, 0.62이면 62에 가까워집니다. 실제 ANN은 이 과정을 수많은 Weight와 Bias에 대해 자동으로 수행합니다.

### 신경망 내부에서 하는 계산

**Dense Layer**

`z = XW + b`

- `X` : 입력값(Features)
- `W` : Weight
- `b` : Bias
- `z` : Dense 계산 결과

**Activation Function**

`a = φ(z) = φ(XW+b)`

Activation Function은 Dense의 선형 계산에 비선형성을 추가합니다. ReLU를 선택하면 `a=max(0,z)`가 됩니다.

**Output Layer**에서 최종 예측값 `ŷ`를 만들고, 실제값 `y`와 비교하여 Loss를 계산합니다. Backpropagation은 어떤 Weight/Bias가 Loss에 얼마나 영향을 주었는지 계산하고, Adam Optimizer가 실제 값을 수정합니다.
        """
    )

with st.expander("📚 자주 나오는 용어 사전", expanded=False):
    glossary = pd.DataFrame(
        [
            ["Feature", "ANN에 넣는 입력변수 x. 예: BMI, 혈압, 혈당"],
            ["Target", "ANN이 맞히려고 하는 출력값 y"],
            ["Input Layer", "Feature를 신경망에 전달하는 입력층"],
            ["Neuron", "입력에 Weight를 곱하고 Bias를 더한 뒤 Activation을 적용하는 계산 단위"],
            ["Dense Layer", "이전 층의 모든 출력과 다음 뉴런을 연결하는 완전연결층. z=XW+b 계산"],
            ["Weight", "각 입력이 다음 뉴런에 미치는 영향의 크기. 학습하면서 바뀜"],
            ["Bias", "Weight 계산만으로 부족한 부분을 보정하는 학습 파라미터"],
            ["Activation", "Dense 결과에 비선형성을 주는 함수. 예: ReLU"],
            ["Loss", "현재 예측이 정답과 얼마나 다른지 나타내는 학습용 오차"],
            ["Backpropagation", "Loss를 줄이기 위해 각 Weight/Bias가 얼마나 수정되어야 하는지 Gradient를 계산"],
            ["Adam", "Gradient를 이용해 Weight/Bias를 실제로 갱신하는 Optimizer"],
            ["Epoch", "Train 데이터 전체를 한 번 학습한 횟수"],
            ["Batch size", "Weight를 한 번 갱신할 때 사용하는 데이터 묶음 크기"],
            ["Train", "Weight/Bias를 직접 학습하는 데이터"],
            ["Validation", "학습 중 과적합 여부와 일반화 성능을 확인하는 데이터"],
            ["Test", "학습 완료 후 최종 성능을 평가하는 데이터"],
            ["Standardization", "Feature의 크기를 맞추는 전처리. x'=(x-평균)/표준편차"],
            ["Overfitting", "Train 데이터는 잘 맞지만 새로운 데이터에는 잘 못 맞는 과적합 상태"],
            ["Underfitting", "모델이 너무 단순하거나 정보가 부족해 Train/Validation 모두 잘 못 맞는 상태"],
        ],
        columns=["용어", "뜻"],
    )
    st.dataframe(glossary, use_container_width=True, hide_index=True)

with st.expander("🧪 실습 권장 순서", expanded=False):
    st.markdown(
        """
1. **실습용 예제 데이터**를 선택합니다.
2. Target은 `Disease_Progression`, Feature는 기본 3개로 먼저 실행합니다.
3. Hidden neurons=3으로 학습하여 작은 ANN의 Weight 개수를 확인합니다.
4. Feature를 더 늘리거나 Hidden neurons를 8→16→32로 바꾸며 Test 성능을 비교합니다.
5. R²가 올라가고 RMSE/MAE가 내려가는지 확인합니다.
6. Learning Curve에서 Train Loss와 Validation Loss의 차이를 보고 과적합 여부를 확인합니다.
7. Weight/Bias 탭에서 실제 학습된 파라미터를 확인합니다.
8. 마지막으로 새 Feature 값을 입력해 예측합니다.

**중요:** R²는 '맞출 확률'이 아닙니다. 예를 들어 R²=0.52는 52% 확률로 맞는다는 뜻이 아니라, Test 데이터의 Target 변동을 모델이 약 52% 설명한다는 의미입니다.
        """
    )


# ============================================================
# HELPERS
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@st.cache_data(show_spinner=False)
def load_example_data() -> pd.DataFrame:
    # scaled=False: 원래 단위에 가까운 값을 보여주기 위함.
    dataset = load_diabetes(as_frame=True, scaled=False)
    df = dataset.frame.copy()

    rename_map = {
        "age": "Age",
        "sex": "Sex_Code",
        "bmi": "BMI",
        "bp": "Blood_Pressure",
        "s1": "Total_Cholesterol",
        "s2": "LDL",
        "s3": "HDL",
        "s4": "Cholesterol_HDL_Ratio",
        "s5": "Log_Triglycerides",
        "s6": "Blood_Sugar",
        "target": "Disease_Progression",
    }
    return df.rename(columns=rename_map)


def read_uploaded_file(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    ext = uploaded.name.lower().rsplit(".", 1)[-1]

    if ext == "csv":
        try:
            return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), encoding="cp949")

    if ext in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(raw))

    raise ValueError("CSV, XLSX, XLS만 지원합니다.")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= max(3, int(len(df) * 0.5)):
            cols.append(col)
    return cols


def build_model(
    input_dim: int,
    hidden_units: int,
    activation_name: str,
    learning_rate: float,
    l2_value: float,
) -> tf.keras.Model:
    model = tf.keras.Sequential(name="ANN_Regressor")

    # Input Layer: 값 전달
    model.add(
        tf.keras.Input(
            shape=(input_dim,),
            name="Input_Layer",
        )
    )

    # Dense Hidden Layer: z = XW + b
    model.add(
        tf.keras.layers.Dense(
            units=hidden_units,
            activation=None,
            use_bias=True,
            kernel_regularizer=(
                tf.keras.regularizers.l2(l2_value)
                if l2_value > 0
                else None
            ),
            name="Dense_Hidden_Layer",
        )
    )

    # Activation Function: a = phi(z)
    model.add(
        tf.keras.layers.Activation(
            activation_name,
            name="Activation_Function",
        )
    )

    # Output Layer: regression -> linear
    model.add(
        tf.keras.layers.Dense(
            units=1,
            activation="linear",
            name="Dense_Output_Layer",
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="mse",
        metrics=["mae"],
    )
    return model


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def model_summary_text(model) -> str:
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


# ============================================================
# 0. ANN 학습 원리 — Weight가 어떻게 바뀌는가?
# ============================================================
st.subheader("0. ANN 학습 원리부터 보기")

with st.expander("🎯 Weight가 오차를 줄이는 원리 — 간단한 1개 뉴런 예시", expanded=False):
    st.markdown(
        """
아래 예시는 복잡한 ANN에 들어가기 전에 **Weight가 왜 학습되는지** 보여주는 직관적인 예시입니다.

입력값 `x = 100 km`, 정답 `y = 62.137 miles`라고 두고,
하나의 Weight `w`로 다음처럼 예측한다고 가정합니다.

`예측값 ŷ = x × w`

실제 ANN에서는 이 과정을 수많은 Weight/Bias에 대해
**Backpropagation + Adam**이 자동으로 수행합니다.
        """
    )

    weight = st.slider(
        "Weight w",
        min_value=0.40,
        max_value=0.80,
        value=0.50,
        step=0.01,
    )

    x_demo = 100.0
    y_true_demo = 62.137
    y_pred_demo = x_demo * weight
    error_demo = y_true_demo - y_pred_demo

    d1, d2, d3 = st.columns(3)
    d1.metric("입력 x", "100 km")
    d2.metric("예측 ŷ = 100 × w", f"{y_pred_demo:.3f} miles")
    d3.metric("Error = y - ŷ", f"{error_demo:+.3f}")

    if abs(error_demo) < 1.5:
        st.success(
            "예측값이 정답에 가까워졌습니다. Weight를 조정하면서 Error가 작아지는 것을 확인할 수 있습니다."
        )
    elif error_demo > 0:
        st.info(
            "예측값이 정답보다 작습니다. 이 단순 예시에서는 Weight를 조금 키우면 Error가 줄어듭니다."
        )
    else:
        st.warning(
            "예측값이 정답보다 큽니다. 이 단순 예시에서는 Weight를 조금 줄이면 Error가 줄어듭니다."
        )

with st.expander("🧩 여러 입력값을 사용하는 ANN 구조", expanded=True):
    st.markdown(
        """
기본 실습은 **입력변수 3개 → Hidden neuron 3개 → Activation → Output 1개** 구조로 시작합니다.

**Input Layer (3)**  
→ **Dense Hidden Layer (3)**  
→ **Activation Function**  
→ **Dense Output Layer (1)**

입력 3개와 Hidden neuron 3개가 모두 연결되면:

`3 × 3 = 9개의 Input→Hidden Weight`

Hidden neuron 3개가 Output 1개와 연결되면:

`3 × 1 = 3개의 Hidden→Output Weight`

Dense Hidden Layer에서는:

`z = XW + b`

Activation Function에서는:

`a = φ(z) = φ(XW + b)`

를 계산합니다.
        """
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ ANN 설정")

    st.subheader("① 데이터")

    data_mode = st.radio(
        "데이터 선택",
        ["실습용 예제 데이터", "CSV / Excel 업로드"],
        help="처음에는 실습용 예제 데이터를 권장합니다.",
    )

    uploaded = None
    if data_mode == "CSV / Excel 업로드":
        uploaded = st.file_uploader(
            "파일 업로드",
            type=["csv", "xlsx", "xls"],
        )

    st.divider()
    st.subheader("④ 기본 ANN 구조")

    hidden_units = st.number_input(
        "Hidden neurons",
        min_value=1,
        max_value=256,
        value=3,
        step=1,
        help="Hidden Layer에 둘 뉴런 개수입니다. 기본값 3은 구조를 이해하기 쉽게 하기 위한 값입니다.",
    )

    activation_label = st.selectbox(
        "Activation Function",
        ["ReLU", "tanh", "sigmoid"],
        index=0,
        help="Dense에서 계산한 z=XW+b를 비선형 값 a=φ(z)로 변환합니다.",
    )

    activation_name = {
        "ReLU": "relu",
        "tanh": "tanh",
        "sigmoid": "sigmoid",
    }[activation_label]

    with st.expander("🔧 고급 설정 — 처음에는 기본값 사용"):
        learning_rate = st.select_slider(
            "Learning rate",
            options=[0.0001, 0.0003, 0.001, 0.003, 0.01],
            value=0.001,
            format_func=lambda x: f"{x:g}",
            help="Adam이 Weight/Bias를 한 번에 얼마나 수정할지 정합니다.",
        )

        l2_value = st.select_slider(
            "L2 Regularization",
            options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2],
            value=1e-4,
            format_func=lambda x: f"{x:g}",
            help="Weight가 지나치게 커지는 것을 억제합니다.",
        )

        epochs = st.number_input(
            "Max epochs",
            min_value=50,
            max_value=3000,
            value=600,
            step=50,
        )

        batch_size = st.selectbox(
            "Batch size",
            [8, 16, 32, 64],
            index=1,
        )

        patience = st.number_input(
            "EarlyStopping patience",
            min_value=5,
            max_value=200,
            value=40,
            step=5,
        )

        validation_ratio = st.slider(
            "Validation ratio",
            0.10,
            0.25,
            0.15,
            0.05,
        )

        test_ratio = st.slider(
            "Test ratio",
            0.10,
            0.30,
            0.20,
            0.05,
        )

        seed = st.number_input(
            "Random seed",
            0,
            9999,
            42,
            1,
        )


# ============================================================
# 1. DATA
# ============================================================
if data_mode == "실습용 예제 데이터":
    df = load_example_data()
    st.info(
        "예제 데이터는 원래 단위에 가까운 값으로 표시됩니다. "
        "ANN에 들어가기 전 표준화는 프로그램 내부에서 별도로 수행합니다."
    )
else:
    if uploaded is None:
        st.warning("왼쪽에서 CSV 또는 Excel 파일을 업로드하세요.")
        st.stop()
    df = read_uploaded_file(uploaded)

st.subheader("1. 데이터 확인")

st.markdown(
    """
여기서는 **이번 ANN이 무엇을 예측할지 정의**합니다.

- **Target** = ANN이 맞혀야 하는 최종 출력값
- **Features** = Target을 예측하기 위해 ANN에 넣는 입력정보

예를 들어 Target을 `Disease_Progression`으로 두고 Features를 `BMI`, `Blood_Pressure`, `Blood_Sugar`로 선택하면,
프로그램은 **세 입력값을 이용해 Disease_Progression을 예측하는 ANN**을 만듭니다.
    """
)


numeric_cols = numeric_columns(df)
if len(numeric_cols) < 2:
    st.error("숫자형 열이 최소 2개 필요합니다.")
    st.stop()

if data_mode == "실습용 예제 데이터":
    default_target = "Disease_Progression"
    default_features = ["BMI", "Blood_Pressure", "Blood_Sugar"]
else:
    default_target = numeric_cols[-1]
    default_features = [c for c in numeric_cols if c != default_target][:3]

left, right = st.columns([1, 2])

with left:
    target_col = st.selectbox(
        "예측할 값 (Target)",
        numeric_cols,
        index=numeric_cols.index(default_target),
        help="ANN이 최종적으로 맞히려는 출력값 y입니다.",
    )

with right:
    candidates = [c for c in numeric_cols if c != target_col]
    usable_defaults = [c for c in default_features if c in candidates]

    feature_cols = st.multiselect(
        "입력 변수 (Features)",
        candidates,
        default=usable_defaults,
        help="ANN에 입력하는 x1, x2, x3 ... 입니다. 기본 실습은 3개를 권장합니다.",
    )

if not feature_cols:
    st.warning("Feature를 하나 이상 선택하세요.")
    st.stop()

work = df[feature_cols + [target_col]].copy()
for col in work.columns:
    work[col] = pd.to_numeric(work[col], errors="coerce")
work = work.dropna().reset_index(drop=True)

a1, a2, a3 = st.columns(3)
a1.metric("Samples", f"{len(work):,}")
a2.metric("Input Features", len(feature_cols))
a3.metric("Target", target_col)

with st.expander("📋 원래 데이터 값 보기 / CSV 다운로드", expanded=True):
    st.dataframe(
        work.head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇ 현재 분석 데이터 CSV",
        data=work.to_csv(index=False).encode("utf-8-sig"),
        file_name="ann_analysis_data.csv",
        mime="text/csv",
    )

if data_mode == "실습용 예제 데이터":
    st.caption(
        "예: BMI=32.1, Blood_Pressure=101, Blood_Sugar=87처럼 원래 값이 보입니다. "
        "표준화된 작은 숫자를 미리 저장해 둔 데이터가 아닙니다."
    )


# ============================================================
# 2. STANDARDIZATION
# ============================================================
st.subheader("2. 표준화(Standardization)")

st.info(
    "표준화는 ANN Layer가 아니라 ANN에 데이터를 넣기 전의 전처리 단계입니다. "
    "Train 데이터의 평균과 표준편차를 기준으로 x' = (x-평균)/표준편차 형태로 변환합니다. "
    "Validation / Test / 새 입력값에도 Train에서 만든 같은 기준을 적용합니다."
)

with st.expander("표준화를 왜 하는지 예시로 보기", expanded=False):
    st.markdown(
        """
예를 들어 입력값이 `BMI=30`, `혈압=120`, `콜레스테롤=220`이라면 숫자의 크기가 서로 다릅니다. ANN은 Gradient를 이용해 Weight를 학습하기 때문에 입력 스케일 차이가 너무 크면 학습이 불안정하거나 느려질 수 있습니다.

StandardScaler는 각 Feature를 다음 식으로 변환합니다.

`x' = (x - Train 평균) / Train 표준편차`

변환 후 `0`은 Train 평균과 비슷한 값, `+1`은 평균보다 약 1 표준편차 큰 값, `-1`은 평균보다 약 1 표준편차 작은 값을 의미합니다. 원래 데이터 자체를 없애는 것이 아니라 **ANN에 입력할 때 사용하는 표현을 바꾸는 것**입니다.

또한 평균과 표준편차는 반드시 Train 데이터에서만 계산합니다. Test 데이터까지 포함해 기준을 만들면 시험 문제의 정보를 미리 본 것과 같은 Data Leakage가 발생할 수 있습니다.
        """
    )

st.markdown(
    """
**왜 필요한가?**

예를 들어 BMI는 약 `20~40`, 혈압은 약 `60~130`, 다른 변수는 수백 단위일 수 있습니다.
값의 크기가 크게 다르면 학습이 불안정해질 수 있으므로 ANN에 넣기 전에 크기를 맞춥니다.
    """
)


# ============================================================
# 3. ARCHITECTURE
# ============================================================
st.subheader("3. ANN 구조 확인")

st.markdown(
    """
이 표는 **현재 사용자가 선택한 설정으로 실제 만들어질 ANN 구조**입니다.
Feature 수가 바뀌면 Input 개수가 바뀌고, Hidden neurons 값을 바꾸면 Dense Hidden Layer의 뉴런 수와 Weight 개수도 자동으로 바뀝니다.
    """
)


input_count = len(feature_cols)
input_hidden_weights = input_count * int(hidden_units)
hidden_output_weights = int(hidden_units) * 1

architecture = pd.DataFrame(
    [
        {
            "순서": 1,
            "Layer": "Input Layer",
            "구성": f"{input_count} Features",
            "계산": "입력값 전달",
        },
        {
            "순서": 2,
            "Layer": "Dense Hidden Layer",
            "구성": f"{hidden_units} neurons",
            "계산": "z = XW + b",
        },
        {
            "순서": 3,
            "Layer": "Activation Function",
            "구성": activation_label,
            "계산": "a = φ(z)",
        },
        {
            "순서": 4,
            "Layer": "Dense Output Layer",
            "구성": "1 neuron / linear",
            "계산": "예측값 ŷ 출력",
        },
    ]
)

st.dataframe(
    architecture,
    use_container_width=True,
    hide_index=True,
)

w1, w2, w3 = st.columns(3)
w1.metric("Input → Hidden Weight", f"{input_hidden_weights}개")
w2.metric("Hidden → Output Weight", f"{hidden_output_weights}개")
w3.metric("Activation", activation_label)

if input_count == 3 and int(hidden_units) == 3:
    st.success(
        "현재 기본 구조는 입력 3개 × Hidden 3개 = Weight 9개, "
        "Hidden 3개 × Output 1개 = Weight 3개 구조입니다."
    )

with st.expander("📖 현재 ANN 구조를 읽는 방법", expanded=False):
    parameter_total = input_count * int(hidden_units) + int(hidden_units) + int(hidden_units) + 1
    st.markdown(
        f"""
현재 선택한 Feature는 **{input_count}개**, Hidden neuron은 **{int(hidden_units)}개**입니다.

- Input→Hidden Weight: `{input_count} × {int(hidden_units)} = {input_count * int(hidden_units)}`개
- Hidden Bias: `{int(hidden_units)}`개
- Hidden→Output Weight: `{int(hidden_units)} × 1 = {int(hidden_units)}`개
- Output Bias: `1`개
- 따라서 총 Trainable Parameter는 **{parameter_total}개**입니다.

뉴런 수를 늘리면 더 복잡한 관계를 표현할 수 있지만, 데이터가 적은데 모델만 너무 커지면 Train 데이터를 외우는 과적합이 생길 수 있습니다. 반대로 뉴런이 너무 적으면 복잡한 관계를 표현하지 못해 과소적합이 생길 수 있습니다. 그래서 3, 8, 16, 32처럼 여러 후보를 같은 데이터 분할에서 비교해 Validation/Test 성능이 좋은 구조를 선택합니다.
        """
    )


# ============================================================
# 4. TRAIN
# ============================================================
st.subheader("4. ANN 학습 및 평가")

st.markdown(
    """
이 단계부터는 **설명 화면이 아니라 실제 모델 학습**이 수행됩니다.
현재 선택한 Target, Features, Hidden neuron, Activation 및 고급 설정을 이용해 새로운 ANN을 생성한 뒤 학습합니다.
    """
)


st.markdown(
    """
버튼을 누르면 다음 과정이 자동으로 실행됩니다.

**Train / Validation / Test 분리**  
→ **Train 기준 표준화**  
→ **Forward Propagation**  
→ **MSE Loss 계산**  
→ **Backpropagation**  
→ **Adam으로 Weight/Bias 갱신**  
→ **반복 학습**  
→ **Test 성능평가**
    """
)

with st.expander("각 학습 단계에서 실제로 무엇을 하는가?", expanded=False):
    st.markdown(
        """
**1) Forward Propagation**  
현재 Weight/Bias를 사용하여 Input에서 Output 방향으로 계산하고 예측값 `ŷ`를 만듭니다.

**2) MSE Loss**  
`MSE = 평균((y-ŷ)²)`로 예측과 정답의 차이를 하나의 숫자로 계산합니다.

**3) Backpropagation**  
Loss를 기준으로 각 Weight/Bias를 조금 바꿨을 때 Loss가 증가하는지 감소하는지 Gradient를 계산합니다.

**4) Adam Optimizer**  
Gradient를 이용해 실제 Weight/Bias 값을 수정합니다. 사람이 Weight를 0.5→0.6처럼 직접 바꾸는 과정을 Adam이 자동으로 수행한다고 생각하면 됩니다.

**5) Epoch 반복**  
Train 데이터 전체를 한 번 학습하면 1 Epoch입니다. 여러 Epoch 동안 위 과정을 반복합니다.

**6) Validation 확인**  
Validation Loss가 더 이상 좋아지지 않으면 과적합을 막기 위해 EarlyStopping이 학습을 종료합니다.

**7) Test 평가**  
마지막에는 학습에 사용하지 않은 Test 데이터로 R², RMSE, MAE를 계산합니다. Test 데이터는 최종 시험이라고 생각하면 됩니다.
        """
    )

if validation_ratio + test_ratio >= 0.50:
    st.error("Validation + Test 비율이 너무 큽니다.")
    st.stop()

if st.button(
    "▶ ANN 학습 및 평가 실행",
    type="primary",
    use_container_width=True,
):
    set_seed(int(seed))

    X = work[feature_cols].to_numpy(dtype=np.float32)
    y = work[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=float(test_ratio),
        random_state=int(seed),
    )

    adjusted_val = float(validation_ratio) / (1.0 - float(test_ratio))

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=adjusted_val,
        random_state=int(seed),
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s = y_scaler.transform(y_val)

    model = build_model(
        input_dim=len(feature_cols),
        hidden_units=int(hidden_units),
        activation_name=activation_name,
        learning_rate=float(learning_rate),
        l2_value=float(l2_value),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(patience),
            restore_best_weights=True,
            min_delta=1e-5,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(5, int(patience) // 3),
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    with st.spinner("Weight와 Bias를 학습하고 있습니다..."):
        history_obj = model.fit(
            X_train_s,
            y_train_s,
            validation_data=(X_val_s, y_val_s),
            epochs=int(epochs),
            batch_size=int(batch_size),
            callbacks=callbacks,
            verbose=0,
        )

    history = history_obj.history

    pred_s = model.predict(X_test_s, verbose=0)
    pred = y_scaler.inverse_transform(pred_s).ravel()
    actual = y_test.ravel()

    ann_metrics = calculate_metrics(actual, pred)

    baseline = LinearRegression()
    baseline.fit(X_train, y_train.ravel())
    base_pred = baseline.predict(X_test)
    base_metrics = calculate_metrics(actual, base_pred)

    results = pd.DataFrame(
        {
            "Actual": actual,
            "ANN_Predicted": pred,
            "Residual": actual - pred,
            "Linear_Baseline": base_pred,
        }
    )

    # 학습 데이터 표준화 전/후 비교 예시
    raw_preview = pd.DataFrame(
        X_train[:5],
        columns=feature_cols,
    )
    scaled_preview = pd.DataFrame(
        X_train_s[:5],
        columns=feature_cols,
    )

    st.session_state["ann_result_v2"] = {
        "model": model,
        "history": history,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "train_raw": X_train,
        "results": results,
        "ann_metrics": ann_metrics,
        "base_metrics": base_metrics,
        "raw_preview": raw_preview,
        "scaled_preview": scaled_preview,
        "split_counts": {
            "Train": len(X_train),
            "Validation": len(X_val),
            "Test": len(X_test),
        },
    }


# ============================================================
# 5. RESULTS
# ============================================================
res = st.session_state.get("ann_result_v2")

if res:
    st.divider()
    st.header("5. 결과 분석")

    m = res["ann_metrics"]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("R²", f"{m['R2']:.4f}")
    r2.metric("RMSE", f"{m['RMSE']:.4f}")
    r3.metric("MAE", f"{m['MAE']:.4f}")
    r4.metric("Trainable Parameters", f"{res['model'].count_params():,}")

    with st.expander("📖 결과 숫자 읽는 법 — R²는 맞출 확률이 아닙니다", expanded=True):
        r2_value = m["R2"]
        explained_text = (
            f"약 {r2_value * 100:.1f}%의 변동을 설명"
            if r2_value >= 0
            else "평균값만 예측하는 기준보다도 성능이 낮음"
        )
        st.markdown(
            f"""
### R² = {r2_value:.4f}

R²는 **적중률이나 확률이 아닙니다.** `R²=0.50`이라고 해서 두 번 중 한 번 맞힌다는 뜻이 아닙니다.

공식은 `R² = 1 - SSE/SST`이며, 현재 Test 데이터에서는 **{explained_text}**한다는 의미로 해석할 수 있습니다. 일반적으로 같은 문제와 같은 Test 데이터에서 더 높은 R²가 더 좋지만, 데이터와 문제 특성에 따라 가능한 수준이 다릅니다.

### RMSE = {m['RMSE']:.4f}

실제값과 예측값의 차이를 제곱해서 평균낸 뒤 제곱근을 취한 값입니다. **Target과 같은 단위**로 표시되고 큰 오차에 더 큰 벌점을 줍니다. 작을수록 좋습니다.

### MAE = {m['MAE']:.4f}

`|실제값-예측값|`의 평균입니다. 현재 모델이 Test 데이터에서 평균적으로 어느 정도 차이를 보이는지 가장 직관적으로 볼 수 있습니다. 작을수록 좋습니다.

### Trainable Parameters = {res['model'].count_params():,}

ANN이 학습을 통해 바꾸는 **Weight와 Bias의 총 개수**입니다. 파라미터가 많으면 복잡한 관계를 표현할 수 있지만, 너무 많으면 과적합 가능성도 커집니다.

**중요:** 좋은 모델은 R² 하나만 높다고 판단하지 않습니다. R², RMSE, MAE, Learning Curve, Residual, Validation 성능을 함께 봅니다.
            """
        )

    tabs = st.tabs(
        [
            "📈 예측 성능",
            "📉 학습 과정",
            "🔄 표준화 확인",
            "🧪 선형모델 비교",
            "🧬 Weight / Bias",
            "🔮 새 데이터 예측",
        ]
    )

    with tabs[0]:
        st.info(
            "Actual vs Predicted의 점들이 y=x 선에 가까울수록 좋습니다. "
            "Residual은 0을 중심으로 특정 패턴 없이 분포하는 것이 이상적입니다."
        )

        rdf = res["results"]
        c1, c2 = st.columns(2)

        with c1:
            lo = float(min(rdf["Actual"].min(), rdf["ANN_Predicted"].min()))
            hi = float(max(rdf["Actual"].max(), rdf["ANN_Predicted"].max()))

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=rdf["Actual"],
                    y=rdf["ANN_Predicted"],
                    mode="markers",
                    name="Test data",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[lo, hi],
                    mode="lines",
                    name="Ideal y=x",
                )
            )
            fig.update_layout(
                title="Actual vs Predicted",
                xaxis_title="Actual",
                yaxis_title="Predicted",
                height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure(
                go.Scatter(
                    x=rdf["ANN_Predicted"],
                    y=rdf["Residual"],
                    mode="markers",
                    name="Residual",
                )
            )
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(
                title="Residual Plot",
                xaxis_title="Predicted",
                yaxis_title="Actual - Predicted",
                height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(rdf, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.info(
            "Train Loss는 실제로 공부한 문제의 오차, Validation Loss는 공부에 직접 쓰지 않은 모의고사의 오차라고 생각하면 됩니다. "
            "둘이 함께 감소하면 일반화 성능도 좋아지고 있는 것이고, Train Loss만 계속 감소하면서 Validation Loss가 올라가면 과적합 가능성이 있습니다."
        )

        hist = res["history"]
        n = len(hist["loss"])
        best_epoch = int(np.argmin(hist["val_loss"])) + 1

        q1, q2 = st.columns(2)
        q1.metric("실제 학습 Epoch", n)
        q2.metric("Best Validation Epoch", best_epoch)

        with st.expander("Epoch 숫자는 어떻게 해석하나?", expanded=False):
            st.markdown(
                f"""
- **실제 학습 Epoch = {n}**: EarlyStopping이 최종적으로 학습을 멈춘 시점입니다.
- **Best Validation Epoch = {best_epoch}**: Validation Loss가 가장 낮았던 시점입니다.
- 프로그램은 `restore_best_weights=True`를 사용하므로 최종 예측에는 마지막 Epoch의 Weight가 아니라 **Best Validation Epoch의 Weight**를 복원해 사용합니다.

예를 들어 Best Epoch가 91이고 실제 종료가 131이라면, 91 이후 일정 기간 더 지켜봤지만 더 좋은 Validation Loss가 나오지 않아 학습을 종료했다는 뜻입니다.
                """
            )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(1, n + 1)),
                y=hist["loss"],
                mode="lines",
                name="Train Loss",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(1, n + 1)),
                y=hist["val_loss"],
                mode="lines",
                name="Validation Loss",
            )
        )
        fig.update_layout(
            title="Learning Curve",
            xaxis_title="Epoch",
            yaxis_title="MSE Loss",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.info(
            "왼쪽은 원래 데이터 값, 오른쪽은 Train 데이터의 평균과 표준편차를 이용해 "
            "ANN 입력용으로 변환한 값입니다. 표준화는 신경망 Layer가 아니라 전처리 단계입니다."
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 표준화 전")
            st.dataframe(
                res["raw_preview"],
                use_container_width=True,
                hide_index=True,
            )

        with c2:
            st.markdown("#### 표준화 후")
            st.dataframe(
                res["scaled_preview"],
                use_container_width=True,
                hide_index=True,
            )

        scaler = res["x_scaler"]
        scaler_info = pd.DataFrame(
            {
                "Feature": res["feature_cols"],
                "Train Mean": scaler.mean_,
                "Train Std": scaler.scale_,
            }
        )
        st.markdown("#### Train 데이터에서 계산한 표준화 기준")
        st.dataframe(scaler_info, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.info(
            "ANN이 실제로 복잡한 비선형 관계를 학습해 이득을 얻었는지 확인하기 위해 동일한 Train/Test 분할에서 Linear Regression과 비교합니다. "
            "ANN이 더 복잡하다는 이유만으로 항상 더 좋은 것은 아닙니다."
        )

        comparison = pd.DataFrame(
            {
                "Model": ["Dense ANN", "Linear Regression"],
                "R²": [res["ann_metrics"]["R2"], res["base_metrics"]["R2"]],
                "RMSE": [res["ann_metrics"]["RMSE"], res["base_metrics"]["RMSE"]],
                "MAE": [res["ann_metrics"]["MAE"], res["base_metrics"]["MAE"]],
            }
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        ann_rmse = res["ann_metrics"]["RMSE"]
        base_rmse = res["base_metrics"]["RMSE"]
        if ann_rmse < base_rmse:
            st.success(
                "현재 Test 데이터에서는 ANN의 RMSE가 Linear Regression보다 낮습니다. "
                "즉 현재 설정에서는 ANN의 비선형 모델링이 예측 성능에 도움이 된 것으로 볼 수 있습니다."
            )
        else:
            st.warning(
                "현재 Test 데이터에서는 Linear Regression의 RMSE가 ANN과 같거나 더 낮습니다. "
                "이 경우 ANN 구조를 바꾸거나, 데이터 관계가 단순한지 확인해야 합니다. 복잡한 모델이 항상 더 좋은 것은 아닙니다."
            )

    with tabs[4]:
        st.info(
            "여기 보이는 값이 ANN이 실제 학습을 통해 찾은 Weight와 Bias입니다. "
            "Weight는 연결의 영향 크기, Bias는 뉴런 출력의 기준점을 조정하는 값입니다. 값 하나만 보고 중요도를 단정하면 안 되며 전체 네트워크와 표준화를 함께 고려해야 합니다."
        )

        hidden = res["model"].get_layer("Dense_Hidden_Layer")
        output = res["model"].get_layer("Dense_Output_Layer")

        hidden_w, hidden_b = hidden.get_weights()
        output_w, output_b = output.get_weights()

        st.markdown("#### Input → Hidden Weight")
        st.caption(
            f"Shape {hidden_w.shape} = {hidden_w.shape[0]} inputs × {hidden_w.shape[1]} hidden neurons"
        )
        st.dataframe(
            pd.DataFrame(
                hidden_w,
                index=res["feature_cols"],
                columns=[f"h{i+1}" for i in range(hidden_w.shape[1])],
            ),
            use_container_width=True,
        )

        st.markdown("#### Hidden Bias")
        st.dataframe(
            pd.DataFrame({"Bias": hidden_b}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Hidden → Output Weight")
        st.dataframe(
            pd.DataFrame(
                output_w,
                index=[f"h{i+1}" for i in range(output_w.shape[0])],
                columns=["Output"],
            ),
            use_container_width=True,
        )

        st.markdown("#### Output Bias")
        st.write(float(output_b[0]))

        with st.expander("모델 전체 구조"):
            st.code(model_summary_text(res["model"]), language=None)

    with tabs[5]:
        st.info(
            "학습된 ANN에 새로운 Feature 값을 입력해 Target을 예측합니다. "
            "Train 데이터 범위를 크게 벗어난 값은 외삽이므로 주의해야 합니다."
        )

        train_df = pd.DataFrame(
            res["train_raw"],
            columns=res["feature_cols"],
        )

        inputs = {}
        cols = st.columns(min(3, len(res["feature_cols"])))

        for i, feature in enumerate(res["feature_cols"]):
            low = float(train_df[feature].min())
            high = float(train_df[feature].max())
            default = float(train_df[feature].median())

            with cols[i % len(cols)]:
                inputs[feature] = st.number_input(
                    feature,
                    value=default,
                    format="%.6g",
                    help=f"Train range: {low:g} ~ {high:g}",
                )

        if st.button("🔮 새 데이터 예측 실행", type="primary"):
            new_x = np.array(
                [[inputs[f] for f in res["feature_cols"]]],
                dtype=np.float32,
            )

            outside = []
            for j, feature in enumerate(res["feature_cols"]):
                low = float(train_df[feature].min())
                high = float(train_df[feature].max())
                value = float(new_x[0, j])
                if value < low or value > high:
                    outside.append(
                        f"{feature}: {value:g} (Train {low:g}~{high:g})"
                    )

            new_s = res["x_scaler"].transform(new_x)
            pred_s = res["model"].predict(new_s, verbose=0)
            pred = res["y_scaler"].inverse_transform(pred_s)[0, 0]

            st.metric(
                f"Predicted {res['target_col']}",
                f"{pred:.4f}",
            )

            if outside:
                st.warning(
                    "Train 범위를 벗어난 입력이 있습니다:\n\n- "
                    + "\n- ".join(outside)
                )
            else:
                st.success("모든 입력값이 Train 데이터 범위 안에 있습니다.")


st.divider()
st.subheader("❓ 자주 헷갈리는 질문")
with st.expander("R²=0.5면 50% 확률로 맞힌다는 뜻인가요?", expanded=False):
    st.markdown("아닙니다. R²는 확률이 아니라 **설명력**입니다. R²=0.5는 Test 데이터의 Target 변동을 모델이 약 50% 설명한다는 의미입니다.")
with st.expander("Hidden neuron을 많이 늘리면 무조건 좋아지나요?", expanded=False):
    st.markdown("아닙니다. 너무 적으면 과소적합, 너무 많으면 과적합이 생길 수 있습니다. 같은 데이터 분할에서 3→8→16→32처럼 비교해 Validation/Test 성능이 좋은 구조를 선택합니다.")
with st.expander("Feature를 많이 넣으면 무조건 좋아지나요?", expanded=False):
    st.markdown("아닙니다. Target과 관련 있는 정보가 늘면 도움이 되지만 불필요한 Feature는 잡음을 늘릴 수 있습니다. Feature 조합도 Validation/Test 성능으로 비교합니다.")
with st.expander("표준화된 값이 실제 측정값인가요?", expanded=False):
    st.markdown("아닙니다. 표준화된 값은 ANN 학습을 위해 크기를 변환한 값입니다. 원래 값은 그대로 보존되며, 프로그램은 Train 기준으로 변환한 값만 신경망 입력에 사용합니다.")
with st.expander("Train Loss가 낮으면 좋은 모델인가요?", expanded=False):
    st.markdown("Train Loss만 낮은 것은 충분하지 않습니다. Validation/Test 성능이 함께 좋아야 합니다. Train만 좋고 Validation이 나쁘면 과적합입니다.")

st.divider()
st.caption(
    "핵심: 원래 데이터 → Train/Validation/Test 분리 → Train 기준 표준화 → "
    "Dense(z=XW+b) → Activation(a=φ(z)) → Output → MSE → "
    "Backpropagation → Adam → Test 평가 → 새 데이터 예측"
)
