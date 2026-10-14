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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler


# ============================================================
# Page
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
    max-width: 1500px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}
[data-testid="stMetric"] {
    border: 1px solid rgba(49, 51, 63, 0.14);
    border-radius: 14px;
    padding: 14px 16px;
}
.flow-box {
    border: 1px solid rgba(49, 51, 63, 0.14);
    border-radius: 14px;
    padding: 16px 18px;
    margin: 8px 0 18px 0;
}
.quick-guide {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
    margin: 12px 0 18px 0;
}
.quick-step {
    border: 1px solid rgba(37, 99, 235, 0.22);
    border-radius: 14px;
    padding: 14px 12px;
    background: rgba(239, 246, 255, 0.70);
    min-height: 92px;
}
.quick-step b {
    display: block;
    margin-bottom: 6px;
}
.quick-step small {
    color: #6B7280;
}
@media (max-width: 1000px) {
    .quick-guide { grid-template-columns: repeat(2, 1fr); }
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧠 ANN Prediction Analyzer")
st.caption(
    "Dense 기반 인공신경망의 구조와 학습 알고리즘을 직접 설정하고, "
    "수치형 데이터를 학습·평가·예측하는 범용 회귀 분석 프로그램"
)

st.markdown(
    """
<div class="flow-box">
<b>ANN 전체 처리 흐름</b><br>
Data → Preprocessing → Train / Validation / Test → Scaling →
Input Layer → Dense Hidden Layer → Activation Function →
Dense Hidden Layer → Activation Function → Linear Output →
Loss → Backpropagation → Adam → Evaluation → Prediction
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("🚀 처음 사용한다면 이 순서대로")
st.markdown(
    """
<div class="quick-guide">
  <div class="quick-step"><b>① 데이터 선택</b><small>예제 데이터 또는 CSV/Excel</small></div>
  <div class="quick-step"><b>② 변수 설정</b><small>Target과 Features 선택</small></div>
  <div class="quick-step"><b>③ ANN 구조 확인</b><small>Hidden Layer와 Activation 확인</small></div>
  <div class="quick-step"><b>④ 학습 실행</b><small>버튼 1번으로 학습·평가</small></div>
  <div class="quick-step"><b>⑤ 결과 분석</b><small>R²·RMSE·MAE·그래프 확인</small></div>
  <div class="quick-step"><b>⑥ 새 값 예측</b><small>새 Feature를 넣어 Target 예측</small></div>
</div>
""",
    unsafe_allow_html=True,
)

st.info(
    "처음 실습할 때는 왼쪽 사이드바의 기본값을 그대로 사용해도 됩니다. "
    "Dropout, L2, Learning rate 같은 항목은 '고급 설정'에 모아두었습니다."
)


# ============================================================
# Helpers
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    dataset = load_diabetes(as_frame=True)
    df = dataset.frame.copy()
    df = df.rename(columns={"target": "Target"})
    return df


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

    raise ValueError("CSV, XLSX, XLS 파일만 지원합니다.")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    result = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= max(3, int(len(df) * 0.5)):
            result.append(col)
    return result


def parse_hidden_layers(text: str) -> tuple[int, ...]:
    values = []

    for token in text.replace(" ", "").split(","):
        if not token:
            continue

        units = int(token)

        if not 1 <= units <= 512:
            raise ValueError("은닉층 뉴런 수는 1~512 범위로 입력하세요.")

        values.append(units)

    if not values:
        raise ValueError("은닉층을 하나 이상 입력하세요.")

    if len(values) > 5:
        raise ValueError("은닉층은 최대 5개까지 지원합니다.")

    return tuple(values)


def activation_name(label: str) -> str:
    return {
        "ReLU": "relu",
        "tanh": "tanh",
        "sigmoid": "sigmoid",
        "ELU": "elu",
    }[label]


def scaler_from_name(name: str):
    if name == "RobustScaler":
        return RobustScaler()
    return StandardScaler()


def build_ann(
    input_dim: int,
    hidden_layers: tuple[int, ...],
    activation: str,
    dropout: float,
    l2_value: float,
    learning_rate: float,
) -> tf.keras.Model:
    model = tf.keras.Sequential(name="ANN_Regressor")

    # Input Layer
    model.add(
        tf.keras.Input(
            shape=(input_dim,),
            name="Input_Layer",
        )
    )

    # Hidden Layers
    for idx, units in enumerate(hidden_layers, start=1):
        # Dense = z = XW + b
        model.add(
            tf.keras.layers.Dense(
                units=units,
                activation=None,
                use_bias=True,
                kernel_regularizer=(
                    tf.keras.regularizers.l2(l2_value)
                    if l2_value > 0
                    else None
                ),
                name=f"Dense_Hidden_{idx}",
            )
        )

        # Activation is separated intentionally
        model.add(
            tf.keras.layers.Activation(
                activation,
                name=f"Activation_{idx}",
            )
        )

        if dropout > 0:
            model.add(
                tf.keras.layers.Dropout(
                    rate=dropout,
                    name=f"Dropout_{idx}",
                )
            )

    # Output Layer for regression
    model.add(
        tf.keras.layers.Dense(
            units=1,
            activation="linear",
            name="Dense_Output",
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


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def architecture_table(
    input_dim: int,
    hidden_layers: tuple[int, ...],
    activation_label: str,
    dropout: float,
) -> pd.DataFrame:
    rows = [
        {
            "Layer": "Input Layer",
            "Configuration": f"{input_dim} features",
            "Role": "입력변수 전달",
        }
    ]

    for idx, units in enumerate(hidden_layers, start=1):
        rows.append(
            {
                "Layer": f"Dense Hidden {idx}",
                "Configuration": f"{units} neurons",
                "Role": "z = XW + b",
            }
        )

        rows.append(
            {
                "Layer": f"Activation {idx}",
                "Configuration": activation_label,
                "Role": "비선형성 부여",
            }
        )

        if dropout > 0:
            rows.append(
                {
                    "Layer": f"Dropout {idx}",
                    "Configuration": f"{dropout:.2f}",
                    "Role": "과적합 완화",
                }
            )

    rows.append(
        {
            "Layer": "Dense Output",
            "Configuration": "1 neuron / linear",
            "Role": "연속형 예측값 출력",
        }
    )

    return pd.DataFrame(rows)


def model_summary_text(model: tf.keras.Model) -> str:
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


def ann_permutation_importance(
    model,
    X_test_scaled,
    y_test_raw,
    y_scaler,
    feature_names,
    repeats=5,
    seed=42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    base_pred_scaled = model.predict(
        X_test_scaled,
        verbose=0,
    )
    base_pred = y_scaler.inverse_transform(
        base_pred_scaled
    ).ravel()

    base_rmse = math.sqrt(
        mean_squared_error(
            y_test_raw,
            base_pred,
        )
    )

    rows = []

    for j, feature in enumerate(feature_names):
        increases = []

        for _ in range(repeats):
            shuffled = X_test_scaled.copy()
            permutation = rng.permutation(len(shuffled))
            shuffled[:, j] = shuffled[permutation, j]

            pred_scaled = model.predict(
                shuffled,
                verbose=0,
            )
            pred = y_scaler.inverse_transform(
                pred_scaled
            ).ravel()

            rmse = math.sqrt(
                mean_squared_error(
                    y_test_raw,
                    pred,
                )
            )

            increases.append(
                max(0.0, rmse - base_rmse)
            )

        rows.append(
            {
                "Feature": feature,
                "Delta_RMSE": float(np.mean(increases)),
            }
        )

    result = pd.DataFrame(rows)
    total = result["Delta_RMSE"].sum()

    if total > 0:
        result["Relative_Importance"] = (
            result["Delta_RMSE"] / total
        )
    else:
        result["Relative_Importance"] = 0.0

    return result.sort_values(
        "Delta_RMSE",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# ANN explanation
# ============================================================
with st.expander("📘 프로그램의 ANN 알고리즘", expanded=True):
    st.markdown(
        """
### 1. Input Layer
선택한 Feature 값이 신경망에 입력됩니다.

### 2. Dense Hidden Layer
각 Dense layer는 Weight와 Bias를 이용해 다음 값을 계산합니다.

`z = XW + b`

### 3. Activation Function
Dense 결과에 활성화 함수를 적용합니다.

`a = φ(z)`

Activation Function은 신경망에 비선형성을 추가합니다.

### 4. Output Layer
회귀 문제이므로 마지막 출력층은 `Dense(1, activation="linear")`로 구성합니다.

### 5. Loss
실제값과 예측값의 평균제곱오차(MSE)를 계산합니다.

### 6. Backpropagation
Loss가 각 Weight와 Bias에 미친 영향을 Gradient로 계산합니다.

### 7. Adam Optimizer
Gradient를 이용해 Weight와 Bias를 갱신합니다.

### 8. 반복 학습 및 검증
Validation Loss를 확인하면서 반복 학습하고,
EarlyStopping과 ReduceLROnPlateau로 과적합과 불필요한 학습을 줄입니다.

### 9. Test Evaluation
학습에 사용하지 않은 Test 데이터에서 R², RMSE, MAE를 계산합니다.

### 10. Prediction
학습된 Weight와 Bias를 이용해 새로운 입력값을 예측합니다.
        """
    )

with st.expander("📚 공개된 구현 참고자료", expanded=False):
    st.markdown(
        """
본 프로그램은 공개된 Keras 공식 문서를 구현 근거로 사용했습니다.

- Keras Dense  
  https://keras.io/api/layers/core_layers/dense/
- Keras Adam  
  https://keras.io/api/optimizers/adam/
- Keras EarlyStopping  
  https://keras.io/api/callbacks/early_stopping/
- Keras ReduceLROnPlateau  
  https://keras.io/api/callbacks/reduce_lr_on_plateau/

예제 데이터는 프로그램 실습을 위한 보조 자료이며,
프로그램의 핵심은 **사용자가 직접 수치형 CSV/Excel 데이터를 입력해 ANN 회귀를 수행하는 것**입니다.
        """
    )


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("⚙️ ANN 설정")
    st.caption(
        "처음에는 기본값 그대로 진행해도 됩니다. "
        "각 항목의 ⓘ 도움말에서 의미를 확인할 수 있습니다."
    )

    st.subheader("① 데이터 입력")

    data_mode = st.radio(
        "데이터 선택",
        ["실습용 예제 데이터", "CSV / Excel 업로드"],
        help=(
            "예제 데이터는 프로그램 사용법을 바로 확인하기 위한 기본 데이터입니다. "
            "본인의 수치형 데이터가 있다면 CSV/Excel 업로드를 선택하세요."
        ),
    )

    uploaded = None

    if data_mode == "CSV / Excel 업로드":
        uploaded = st.file_uploader(
            "데이터 파일 업로드",
            type=["csv", "xlsx", "xls"],
            help="숫자형 입력변수(Feature)와 예측할 값(Target)이 포함된 파일을 업로드하세요.",
        )

    st.divider()
    st.subheader("② 기본 신경망 설정")

    hidden_text = st.text_input(
        "Hidden layers",
        value="64,32",
        help=(
            "은닉층의 뉴런 수를 쉼표로 구분합니다. "
            "예: 64,32 = 첫 번째 은닉층 64개 뉴런 + 두 번째 은닉층 32개 뉴런."
        ),
    )

    activation_label = st.selectbox(
        "Activation Function",
        ["ReLU", "tanh", "sigmoid", "ELU"],
        index=0,
        help=(
            "Dense의 가중합 z=XW+b에 적용하는 비선형 함수입니다. "
            "처음 실습에는 ReLU를 권장합니다."
        ),
    )

    st.success(
        "권장 기본값: Hidden layers = 64,32 / Activation = ReLU"
    )

    with st.expander("🔧 고급 설정 — 처음에는 기본값 권장", expanded=False):
        dropout = st.slider(
            "Dropout",
            0.00,
            0.50,
            0.10,
            0.05,
            help=(
                "학습 중 일부 뉴런 출력을 임시로 제외해 과적합을 줄입니다. "
                "0.10은 10%를 의미합니다."
            ),
        )

        l2_value = st.select_slider(
            "L2 Regularization",
            options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2],
            value=1e-4,
            format_func=lambda x: f"{x:g}",
            help="Weight가 지나치게 커지는 것을 억제하는 규제입니다.",
        )

        learning_rate = st.select_slider(
            "Learning rate",
            options=[0.0001, 0.0003, 0.001, 0.003, 0.01],
            value=0.001,
            format_func=lambda x: f"{x:g}",
            help=(
                "Adam이 Weight/Bias를 한 번에 얼마나 수정할지 정하는 값입니다. "
                "너무 크면 불안정하고 너무 작으면 학습이 느립니다."
            ),
        )

        scaler_name = st.selectbox(
            "Scaler",
            ["StandardScaler", "RobustScaler"],
            help=(
                "입력변수의 크기를 맞춰 ANN이 안정적으로 학습하도록 합니다. "
                "일반적인 경우 StandardScaler를 사용하면 됩니다."
            ),
        )

        epochs = st.number_input(
            "Max epochs",
            50,
            3000,
            700,
            50,
            help="학습을 반복할 최대 횟수입니다. EarlyStopping으로 더 일찍 종료될 수 있습니다.",
        )

        batch_size = st.selectbox(
            "Batch size",
            [8, 16, 32, 64],
            index=1,
            help="한 번의 Weight 업데이트에 사용할 데이터 묶음 크기입니다.",
        )

        patience = st.number_input(
            "EarlyStopping patience",
            5,
            200,
            40,
            5,
            help=(
                "Validation Loss가 이 횟수 동안 좋아지지 않으면 학습을 종료합니다. "
                "과적합과 불필요한 반복을 줄입니다."
            ),
        )

        validation_ratio = st.slider(
            "Validation ratio",
            0.10,
            0.25,
            0.15,
            0.05,
            help="학습 중 모델 상태를 확인하기 위해 따로 보관할 데이터 비율입니다.",
        )

        test_ratio = st.slider(
            "Test ratio",
            0.10,
            0.30,
            0.20,
            0.05,
            help="학습이 끝난 뒤 최종 성능을 평가하는 데이터 비율입니다.",
        )

        seed = st.number_input(
            "Random seed",
            0,
            9999,
            42,
            1,
            help="같은 조건에서 데이터 분할과 초기화를 재현하기 위한 값입니다.",
        )

# ============================================================
# Data load
# ============================================================
if data_mode == "실습용 예제 데이터":
    df = load_sample_data()
    st.info(
        "실습용 예제 데이터가 자동으로 불러와졌습니다. "
        "실제 사용 시에는 CSV/Excel 업로드를 선택하면 됩니다."
    )
else:
    if uploaded is None:
        st.warning(
            "왼쪽 사이드바에서 CSV 또는 Excel 파일을 업로드하세요."
        )
        st.stop()

    df = read_uploaded_file(uploaded)


# ============================================================
# Data setup
# ============================================================
st.subheader("1. 데이터 설정")

numeric_cols = numeric_columns(df)

if len(numeric_cols) < 2:
    st.error("수치형으로 사용할 수 있는 열이 최소 2개 필요합니다.")
    st.stop()

left, right = st.columns([1, 2])

with left:
    default_target = (
        numeric_cols.index("Target")
        if "Target" in numeric_cols
        else len(numeric_cols) - 1
    )

    target_col = st.selectbox(
        "예측할 값 (Target)",
        numeric_cols,
        index=default_target,
        help="ANN이 최종적으로 예측해야 하는 출력값 y를 선택합니다.",
    )

with right:
    feature_candidates = [
        col for col in numeric_cols
        if col != target_col
    ]

    feature_cols = st.multiselect(
        "입력 변수 (Features)",
        feature_candidates,
        default=feature_candidates,
        help="Target을 예측하기 위해 ANN에 입력할 변수 x1, x2, ... 를 선택합니다.",
    )

if not feature_cols:
    st.warning("Feature를 하나 이상 선택하세요.")
    st.stop()

work = df[
    feature_cols + [target_col]
].copy()

for col in work.columns:
    work[col] = pd.to_numeric(
        work[col],
        errors="coerce",
    )

before_rows = len(work)
work = work.dropna().reset_index(drop=True)
removed_rows = before_rows - len(work)

if len(work) < 30:
    st.warning(
        "유효 데이터가 30개 미만입니다. ANN 평가가 불안정할 수 있습니다."
    )

m1, m2, m3, m4 = st.columns(4)
m1.metric("Samples", f"{len(work):,}")
m2.metric("Features", len(feature_cols))
m3.metric("Target", target_col)
m4.metric("Removed rows", removed_rows)

with st.expander("데이터 확인 / 다운로드"):
    st.dataframe(
        work.head(100),
        use_container_width=True,
        height=340,
    )

    st.download_button(
        "⬇ 현재 분석 데이터 CSV",
        data=work.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="ann_analysis_data.csv",
        mime="text/csv",
    )


# ============================================================
# Architecture
# ============================================================
try:
    hidden_layers = parse_hidden_layers(hidden_text)
except Exception as exc:
    st.error(str(exc))
    st.stop()

activation = activation_name(activation_label)

architecture = architecture_table(
    input_dim=len(feature_cols),
    hidden_layers=hidden_layers,
    activation_label=activation_label,
    dropout=float(dropout),
)

st.subheader("2. ANN 구조 확인")

st.dataframe(
    architecture,
    use_container_width=True,
    hide_index=True,
)

st.info(
    "표를 위에서 아래로 읽으면 현재 ANN의 실제 계산 순서입니다. "
    "Dense Layer는 z = XW + b를 계산하고, Activation Layer는 a = φ(z)를 계산합니다."
)


# ============================================================
# Train
# ============================================================
st.subheader("3. ANN 학습")

st.info(
    "아래 버튼을 누르면 데이터 분리 → Scaling → 순전파 → MSE 계산 → "
    "Backpropagation → Adam으로 Weight/Bias 갱신 → Test 평가까지 자동으로 수행합니다."
)

if validation_ratio + test_ratio >= 0.50:
    st.error("Validation + Test 비율이 너무 큽니다.")
    st.stop()

if st.button(
    "▶ ④ ANN 학습 및 평가 실행",
    type="primary",
    use_container_width=True,
):
    set_seed(int(seed))

    X = work[feature_cols].to_numpy(
        dtype=np.float32
    )

    y = work[target_col].to_numpy(
        dtype=np.float32
    ).reshape(-1, 1)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=float(test_ratio),
        random_state=int(seed),
    )

    adjusted_val = (
        float(validation_ratio)
        / (1.0 - float(test_ratio))
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=adjusted_val,
        random_state=int(seed),
    )

    x_scaler = scaler_from_name(
        scaler_name
    )
    y_scaler = StandardScaler()

    # fit only on train data
    X_train_s = x_scaler.fit_transform(
        X_train
    )
    X_val_s = x_scaler.transform(
        X_val
    )
    X_test_s = x_scaler.transform(
        X_test
    )

    y_train_s = y_scaler.fit_transform(
        y_train
    )
    y_val_s = y_scaler.transform(
        y_val
    )

    model = build_ann(
        input_dim=len(feature_cols),
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=float(dropout),
        l2_value=float(l2_value),
        learning_rate=float(learning_rate),
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

    with st.spinner("ANN을 학습하고 있습니다..."):
        history_obj = model.fit(
            X_train_s,
            y_train_s,
            validation_data=(
                X_val_s,
                y_val_s,
            ),
            epochs=int(epochs),
            batch_size=int(batch_size),
            callbacks=callbacks,
            verbose=0,
        )

    history = history_obj.history

    pred_scaled = model.predict(
        X_test_s,
        verbose=0,
    )

    ann_pred = y_scaler.inverse_transform(
        pred_scaled
    ).ravel()

    y_test_raw = y_test.ravel()

    ann_metrics = metrics(
        y_test_raw,
        ann_pred,
    )

    # Simple baseline for context
    linear = LinearRegression()
    linear.fit(
        X_train,
        y_train.ravel(),
    )
    linear_pred = linear.predict(
        X_test
    )
    linear_metrics = metrics(
        y_test_raw,
        linear_pred,
    )

    result_df = pd.DataFrame(
        {
            "Actual": y_test_raw,
            "ANN_Predicted": ann_pred,
            "Residual": y_test_raw - ann_pred,
            "Linear_Baseline": linear_pred,
        }
    )

    importance_df = ann_permutation_importance(
        model=model,
        X_test_scaled=X_test_s.copy(),
        y_test_raw=y_test_raw,
        y_scaler=y_scaler,
        feature_names=feature_cols,
        repeats=5,
        seed=int(seed),
    )

    st.session_state["ann_result"] = {
        "model": model,
        "history": history,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "X_train_raw": X_train,
        "result_df": result_df,
        "ann_metrics": ann_metrics,
        "linear_metrics": linear_metrics,
        "importance_df": importance_df,
        "architecture": architecture,
        "analysis_data": work,
        "split_counts": {
            "Train": len(X_train),
            "Validation": len(X_val),
            "Test": len(X_test),
        },
        "settings": {
            "hidden_layers": list(hidden_layers),
            "activation": activation_label,
            "dropout": float(dropout),
            "l2": float(l2_value),
            "learning_rate": float(learning_rate),
            "scaler": scaler_name,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "validation_ratio": float(validation_ratio),
            "test_ratio": float(test_ratio),
            "seed": int(seed),
        },
    }


# ============================================================
# Results
# ============================================================
res = st.session_state.get("ann_result")

if res:
    st.divider()
    st.header("4. ANN 결과 분석")

    ann_m = res["ann_metrics"]
    linear_m = res["linear_metrics"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R²", f"{ann_m['R2']:.4f}")
    k2.metric("RMSE", f"{ann_m['RMSE']:.4f}")
    k3.metric("MAE", f"{ann_m['MAE']:.4f}")
    k4.metric(
        "Trainable Parameters",
        f"{res['model'].count_params():,}",
    )

    split = res["split_counts"]

    st.caption(
        f"Train {split['Train']} / "
        f"Validation {split['Validation']} / "
        f"Test {split['Test']} · "
        "최종 평가는 학습에 사용하지 않은 Test 데이터 기준"
    )

    with st.expander("📖 결과 숫자 읽는 법", expanded=True):
        st.markdown(
            """
- **R²**: 모델이 실제값의 변화를 얼마나 설명하는지 나타냅니다. 일반적으로 **1에 가까울수록 좋습니다.**
- **RMSE**: 예측 오차의 크기를 나타내며 큰 오차에 더 민감합니다. **작을수록 좋습니다.**
- **MAE**: 실제값과 예측값의 평균 절대오차입니다. **작을수록 좋습니다.**
- **Trainable Parameters**: ANN이 학습해야 하는 Weight와 Bias의 총 개수입니다. 많다고 무조건 좋은 것은 아닙니다.
        """
        )

    tabs = st.tabs(
        [
            "📈 예측 성능",
            "📉 학습 과정",
            "🧪 모델 비교",
            "🧭 변수 중요도",
            "🧬 ANN 내부",
            "🔮 새 데이터 예측",
            "💾 결과 저장",
        ]
    )

    with tabs[0]:
        st.info(
            "Actual vs Predicted에서는 점들이 대각선 y=x에 가까울수록 예측이 잘 된 것입니다. "
            "Residual Plot에서는 점들이 0선을 중심으로 무작위로 퍼질수록 좋습니다."
        )
        rdf = res["result_df"]

        c1, c2 = st.columns(2)

        with c1:
            lo = float(
                min(
                    rdf["Actual"].min(),
                    rdf["ANN_Predicted"].min(),
                )
            )
            hi = float(
                max(
                    rdf["Actual"].max(),
                    rdf["ANN_Predicted"].max(),
                )
            )

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
                height=500,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with c2:
            fig = go.Figure(
                go.Scatter(
                    x=rdf["ANN_Predicted"],
                    y=rdf["Residual"],
                    mode="markers",
                    name="Residual",
                )
            )

            fig.add_hline(
                y=0,
                line_dash="dash",
            )

            fig.update_layout(
                title="Residual Plot",
                xaxis_title="Predicted",
                yaxis_title="Actual - Predicted",
                height=500,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.dataframe(
            rdf,
            use_container_width=True,
            hide_index=True,
        )

    with tabs[1]:
        st.info(
            "Train Loss와 Validation Loss가 함께 감소하면 정상적으로 학습되는 것입니다. "
            "Train Loss는 계속 감소하는데 Validation Loss가 다시 증가하면 과적합 가능성이 있습니다."
        )
        history = res["history"]
        n_epochs = len(history["loss"])
        best_epoch = int(
            np.argmin(history["val_loss"])
        ) + 1

        c1, c2, c3 = st.columns(3)
        c1.metric("Epochs Run", n_epochs)
        c2.metric("Best Validation Epoch", best_epoch)
        c3.metric(
            "Best Validation Loss",
            f"{min(history['val_loss']):.5f}",
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(1, n_epochs + 1)),
                y=history["loss"],
                mode="lines",
                name="Train Loss",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(1, n_epochs + 1)),
                y=history["val_loss"],
                mode="lines",
                name="Validation Loss",
            )
        )

        fig.update_layout(
            title="Learning Curve",
            xaxis_title="Epoch",
            yaxis_title="MSE Loss",
            height=520,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tabs[2]:
        st.info(
            "ANN과 Linear Regression을 같은 Test 데이터에서 비교합니다. "
            "R²는 높을수록, RMSE와 MAE는 낮을수록 좋은 결과입니다."
        )
        compare_df = pd.DataFrame(
            {
                "Model": [
                    "Dense ANN",
                    "Linear Regression",
                ],
                "R2": [
                    ann_m["R2"],
                    linear_m["R2"],
                ],
                "RMSE": [
                    ann_m["RMSE"],
                    linear_m["RMSE"],
                ],
                "MAE": [
                    ann_m["MAE"],
                    linear_m["MAE"],
                ],
            }
        )

        st.dataframe(
            compare_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Linear Regression은 ANN이 실제로 더 나은 예측을 제공하는지 확인하기 위한 기준 모델입니다."
        )

    with tabs[3]:
        st.info(
            "Feature Importance는 특정 입력변수를 섞었을 때 예측 성능이 얼마나 나빠지는지로 계산합니다. "
            "값이 클수록 현재 모델이 그 Feature에 더 민감하다는 뜻이며 인과관계를 의미하지는 않습니다."
        )
        imp = res["importance_df"].copy()

        fig = go.Figure(
            go.Bar(
                x=imp["Relative_Importance"],
                y=imp["Feature"],
                orientation="h",
            )
        )

        fig.update_layout(
            title="Permutation Feature Importance",
            xaxis_title="Relative Importance",
            yaxis_title="Feature",
            yaxis={"categoryorder": "total ascending"},
            height=max(400, 42 * len(imp)),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        imp["Importance (%)"] = (
            imp["Relative_Importance"] * 100
        )

        st.dataframe(
            imp[
                [
                    "Feature",
                    "Delta_RMSE",
                    "Importance (%)",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Test 데이터에서 특정 Feature를 섞었을 때 RMSE가 얼마나 증가하는지를 이용한 참고용 중요도입니다."
        )

    with tabs[4]:
        st.info(
            "ANN 내부에서는 실제로 학습된 Dense Layer의 Weight와 Bias를 확인할 수 있습니다. "
            "Weight Matrix의 크기는 이전 Layer 뉴런 수 × 다음 Layer 뉴런 수입니다."
        )
        st.dataframe(
            res["architecture"],
            use_container_width=True,
            hide_index=True,
        )

        st.code(
            model_summary_text(
                res["model"]
            ),
            language=None,
        )

        st.markdown("#### Dense Weight / Bias")

        for layer in res["model"].layers:
            if isinstance(
                layer,
                tf.keras.layers.Dense,
            ):
                weights = layer.get_weights()

                if len(weights) == 2:
                    kernel, bias = weights

                    with st.expander(
                        f"{layer.name} · Weight {kernel.shape} · Bias {bias.shape}"
                    ):
                        st.markdown("**Weight Matrix**")
                        st.dataframe(
                            pd.DataFrame(kernel),
                            use_container_width=True,
                        )

                        st.markdown("**Bias Vector**")
                        st.dataframe(
                            pd.DataFrame({"Bias": bias}),
                            use_container_width=True,
                        )

    with tabs[5]:
        st.info(
            "학습이 끝난 모델에 새로운 Feature 값을 넣어 Target을 예측합니다. "
            "Train 범위를 벗어난 값은 외삽이므로 신뢰도가 낮을 수 있어 경고를 표시합니다."
        )
        st.markdown("### 새로운 입력값 예측")

        train_df = pd.DataFrame(
            res["X_train_raw"],
            columns=res["feature_cols"],
        )

        values = {}
        ui_cols = st.columns(
            min(3, len(res["feature_cols"]))
        )

        for idx, feature in enumerate(
            res["feature_cols"]
        ):
            low = float(train_df[feature].min())
            high = float(train_df[feature].max())
            default = float(train_df[feature].median())

            with ui_cols[idx % len(ui_cols)]:
                values[feature] = st.number_input(
                    feature,
                    value=default,
                    format="%.8g",
                    help=f"Train range: {low:g} ~ {high:g}",
                )

        if st.button(
            "🔮 새 데이터 예측 실행",
            type="primary",
        ):
            new_x = np.array(
                [[
                    values[f]
                    for f
                    in res["feature_cols"]
                ]],
                dtype=np.float32,
            )

            outside = []

            for j, feature in enumerate(
                res["feature_cols"]
            ):
                low = float(train_df[feature].min())
                high = float(train_df[feature].max())
                value = float(new_x[0, j])

                if value < low or value > high:
                    outside.append(
                        f"{feature}: {value:g} "
                        f"(Train {low:g}~{high:g})"
                    )

            new_scaled = res["x_scaler"].transform(
                new_x
            )

            pred_scaled = res["model"].predict(
                new_scaled,
                verbose=0,
            )

            pred = res["y_scaler"].inverse_transform(
                pred_scaled
            )[0, 0]

            st.metric(
                f"Predicted {res['target_col']}",
                f"{pred:.6g}",
            )

            if outside:
                st.warning(
                    "일부 입력값이 Train 데이터 범위를 벗어났습니다. "
                    "외삽 예측은 신뢰도가 낮을 수 있습니다.\n\n- "
                    + "\n- ".join(outside)
                )
            else:
                st.success(
                    "모든 입력값이 Train 데이터 범위 안에 있습니다."
                )

    with tabs[6]:
        st.info(
            "분석 결과 Excel에는 Test 예측값·평가지표·ANN 구조·Feature Importance가 저장되고, "
            "JSON에는 모델 설정값이 저장됩니다."
        )
        settings = {
            "target": res["target_col"],
            "features": res["feature_cols"],
            "model_settings": res["settings"],
            "ann_metrics": res["ann_metrics"],
            "linear_baseline_metrics": res["linear_metrics"],
        }

        json_bytes = json.dumps(
            settings,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        excel_buffer = io.BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl",
        ) as writer:
            res["result_df"].to_excel(
                writer,
                sheet_name="Test Predictions",
                index=False,
            )

            pd.DataFrame(
                [
                    {
                        "Model": "Dense ANN",
                        **res["ann_metrics"],
                    },
                    {
                        "Model": "Linear Regression",
                        **res["linear_metrics"],
                    },
                ]
            ).to_excel(
                writer,
                sheet_name="Metrics",
                index=False,
            )

            res["importance_df"].to_excel(
                writer,
                sheet_name="Feature Importance",
                index=False,
            )

            res["architecture"].to_excel(
                writer,
                sheet_name="ANN Architecture",
                index=False,
            )

            res["analysis_data"].to_excel(
                writer,
                sheet_name="Analysis Data",
                index=False,
            )

        c1, c2 = st.columns(2)

        with c1:
            st.download_button(
                "⬇ 분석 결과 Excel",
                data=excel_buffer.getvalue(),
                file_name="ann_analysis_results.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
            )

        with c2:
            st.download_button(
                "⬇ 모델 설정 JSON",
                data=json_bytes,
                file_name="ann_model_config.json",
                mime="application/json",
                use_container_width=True,
            )


st.divider()
st.caption(
    "ANN은 데이터의 비선형 관계를 학습할 수 있지만 항상 단순 모델보다 우수한 것은 아닙니다. "
    "R², RMSE, MAE, Residual, Validation Loss를 함께 확인하세요."
)
