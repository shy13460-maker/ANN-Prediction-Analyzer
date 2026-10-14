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


# ============================================================
# 4. TRAIN
# ============================================================
st.subheader("4. ANN 학습 및 평가")

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

    with st.expander("📖 결과 숫자 읽는 법", expanded=True):
        st.markdown(
            """
- **R²**: 실제값의 변화를 얼마나 설명하는지 나타냅니다. 일반적으로 **1에 가까울수록 좋습니다.**
- **RMSE**: 큰 오차에 더 민감한 평균 예측오차입니다. **작을수록 좋습니다.**
- **MAE**: 실제값과 예측값의 평균 절대오차입니다. **작을수록 좋습니다.**
- **Trainable Parameters**: ANN이 학습하는 Weight와 Bias의 총 개수입니다.
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
            "Train Loss와 Validation Loss가 함께 감소하면 학습이 정상적으로 진행되는 것입니다. "
            "Train Loss는 내려가는데 Validation Loss가 다시 올라가면 과적합 가능성이 있습니다."
        )

        hist = res["history"]
        n = len(hist["loss"])
        best_epoch = int(np.argmin(hist["val_loss"])) + 1

        q1, q2 = st.columns(2)
        q1.metric("실제 학습 Epoch", n)
        q2.metric("Best Validation Epoch", best_epoch)

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
            "ANN이 실제로 비선형 관계를 더 잘 학습했는지 확인하기 위해 "
            "동일한 Test 데이터에서 Linear Regression과 비교합니다."
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

    with tabs[4]:
        st.info(
            "Dense Layer에 실제로 학습된 Weight와 Bias입니다. "
            "Input→Hidden Weight Matrix와 Hidden→Output Weight Matrix를 직접 확인할 수 있습니다."
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
st.caption(
    "핵심: 원래 데이터 → Train/Validation/Test 분리 → Train 기준 표준화 → "
    "Dense(z=XW+b) → Activation(a=φ(z)) → Output → MSE → "
    "Backpropagation → Adam → Test 평가 → 새 데이터 예측"
)
