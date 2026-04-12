import os
import sys
import logging
from contextlib import asynccontextmanager
from enum import Enum
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WINDOW_SIZE = 288
N_CHANNELS = 2
OUTPUT_DIM = 2

WEIGHT_FILES = {
    "lstm": os.path.join(
        ROOT, "saved_models", "MULTIBILSTMcpu-w288-h2-ts800000.weights.h5"
    ),
    "lstmd": os.path.join(
        ROOT, "saved_models", "MULTIBILSTMDcpu-w288-h1-ts800000.weights.h5"
    ),
    "hbnn": os.path.join(
        ROOT, "saved_models", "MULTIBIHBNN-cpu-gc19_a-w288-h20000.weights.h5"
    ),
}

PARAMS = {
    "first_conv_dim": 16,
    "first_conv_kernel": 3,
    "first_conv_activation": "relu",
    "cnn_layers": 1,
    "second_lstm_dim": 32,
    "first_dense_dim": 8,
    "first_dense_activation": "relu",
    "mlp_units": [32, 32],
    "optimizer": "adam",
    "lr": 1e-3,
    "decay": 1e-4,
}

_model_cache: dict = {}


def _build_lstm(weight_file: str):
    """
    Build the LSTM architecture and load weights.

    Avoids calling the class's load_model() because that calls
    tf.keras.backend.clear_session(), which would destroy any models that
    were already built and cached.
    """
    import tf_keras as keras
    from tf_keras.layers import Input, Lambda, Conv1D, LSTM, Dense
    from tf_keras import Model
    from tf_keras.optimizers.legacy import Adam

    input_shape = (WINDOW_SIZE, N_CHANNELS)
    inp = Input(shape=input_shape)
    x = Lambda(lambda z: z)(inp)
    for _ in range(PARAMS["cnn_layers"]):
        x = Conv1D(
            filters=PARAMS["first_conv_dim"],
            kernel_size=PARAMS["first_conv_kernel"],
            strides=1,
            padding="causal",
            activation=PARAMS["first_conv_activation"],
        )(x)
    x = LSTM(PARAMS["second_lstm_dim"])(x)
    for units in PARAMS["mlp_units"]:
        x = Dense(units, activation="relu")(x)
    x = Dense(PARAMS["first_dense_dim"], activation=PARAMS["first_dense_activation"])(x)
    outputs = Dense(OUTPUT_DIM)(x)

    model = Model(inputs=inp, outputs=outputs)
    model.compile(
        loss="mean_squared_error",
        optimizer=Adam(learning_rate=PARAMS["lr"], decay=PARAMS["decay"]),
    )
    model.load_weights(weight_file)
    return model


def _build_lstmd(weight_file: str):
    """
    Build the LSTMD architecture (LSTM + IndependentNormal output) and load weights.
    Returns the keras model directly; calling model(X) yields a TFP distribution.
    """
    import tf_keras as keras
    import tensorflow_probability as tfp
    from tf_keras.layers import Input, Lambda, Conv1D, LSTM, Dense
    from tf_keras import Model
    from tf_keras.optimizers.legacy import Adam

    input_shape = (WINDOW_SIZE, N_CHANNELS)
    inp = Input(shape=input_shape)
    x = Lambda(lambda z: z)(inp)
    for _ in range(PARAMS["cnn_layers"]):
        x = Conv1D(
            filters=PARAMS["first_conv_dim"],
            kernel_size=PARAMS["first_conv_kernel"],
            strides=1,
            padding="causal",
            activation=PARAMS["first_conv_activation"],
        )(x)
    x = LSTM(PARAMS["second_lstm_dim"])(x)
    for units in PARAMS["mlp_units"]:
        x = Dense(units, activation="relu")(x)
    x = Dense(PARAMS["first_dense_dim"], activation=PARAMS["first_dense_activation"])(x)
    dist = Dense(2 * OUTPUT_DIM)(x)
    out = tfp.layers.IndependentNormal(OUTPUT_DIM)(dist)

    model = Model(inputs=inp, outputs=out)

    def _nll(targets, estimated_dist):
        return -estimated_dist.log_prob(targets)

    model.compile(
        loss=_nll,
        optimizer=Adam(learning_rate=PARAMS["lr"], decay=PARAMS["decay"]),
    )
    model.load_weights(weight_file)
    return model


def _build_hbnn(weight_file: str):
    """
    Build the HBNN architecture (LSTM + DenseVariational + IndependentNormal) and load weights.
    kl_weight uses the approximate training-set size (~80 000) to match the saved weights.
    """
    import tensorflow as tf
    import tensorflow_probability as tfp
    import tf_keras as keras
    from tf_keras.layers import Input, Lambda, Conv1D, LSTM, Dense
    from tf_keras import Model
    from tf_keras.optimizers.legacy import Adam
    from models.HBNN import VarLayer, HBNNPredictor

    KL_WEIGHT = 1.0 / 80_000

    hbnn = HBNNPredictor()

    input_shape = (WINDOW_SIZE, N_CHANNELS)
    inp = Input(shape=input_shape)
    x = Lambda(lambda z: z)(inp)
    for _ in range(PARAMS["cnn_layers"]):
        x = Conv1D(
            filters=PARAMS["first_conv_dim"],
            kernel_size=PARAMS["first_conv_kernel"],
            strides=1,
            padding="causal",
            activation=PARAMS["first_conv_activation"],
        )(x)
    x = LSTM(PARAMS["second_lstm_dim"])(x)
    for units in PARAMS["mlp_units"]:
        x = Dense(units, activation="relu")(x)
    x = VarLayer(
        name="var",
        units=PARAMS["first_dense_dim"],
        make_posterior_fn=hbnn.posterior,
        make_prior_fn=hbnn.prior,
        kl_weight=KL_WEIGHT,
        activation=PARAMS["first_dense_activation"],
    )(x)
    dist = Dense(2 * OUTPUT_DIM)(x)
    out = tfp.layers.IndependentNormal(OUTPUT_DIM)(dist)

    model = Model(inputs=inp, outputs=out)

    def _nll(targets, estimated_dist):
        return -estimated_dist.log_prob(targets)

    model.compile(
        loss=_nll,
        optimizer=Adam(learning_rate=PARAMS["lr"], decay=PARAMS["decay"]),
    )
    model.load_weights(weight_file)
    return model


def _load_model(model_name: str):
    """Build architecture and load weights for the given model name."""
    weight_file = WEIGHT_FILES[model_name]
    if not os.path.exists(weight_file):
        raise FileNotFoundError(
            f"Weight file not found for '{model_name}': {weight_file}. "
            "Run the corresponding *_training.py script first."
        )
    logger.info("Loading '%s' from %s …", model_name, weight_file)
    builders = {"lstm": _build_lstm, "lstmd": _build_lstmd, "hbnn": _build_hbnn}
    model = builders[model_name](weight_file)
    logger.info("'%s' loaded and cached.", model_name)
    return model


def get_model(model_name: str):
    """Return the cached keras model, loading it on first access."""
    if model_name not in _model_cache:
        try:
            _model_cache[model_name] = _load_model(model_name)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to load model '{model_name}': {exc}") from exc
    return _model_cache[model_name]


def _minmax_scale(arr: np.ndarray) -> np.ndarray:
    """Map raw [0, 1] utilisation to the [-1, 1] training scale."""
    return arr * 2.0 - 1.0


def _minmax_inverse(arr: np.ndarray) -> np.ndarray:
    """Invert [-1, 1] model output back to [0, 1] utilisation."""
    return (arr + 1.0) / 2.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload every model whose weights file exists at startup."""
    for name in ("lstm", "lstmd", "hbnn"):
        if os.path.exists(WEIGHT_FILES[name]):
            try:
                get_model(name)
            except Exception as exc:
                logger.warning("Could not preload '%s': %s", name, exc)
        else:
            logger.warning(
                "Skipping preload of '%s' — weight file not found: %s",
                name,
                WEIGHT_FILES[name],
            )
    yield
    _model_cache.clear()


app = FastAPI(
    title="VCC: Uncertainty-Aware Workload Prediction API",
    description=(
        "Predict cloud workload (CPU & Memory utilisation) using three deep-learning models.\n\n"
        "## Models\n"
        "| Model  | Output | Uncertainty |\n"
        "|--------|--------|-------------|\n"
        "| `lstm` | Point estimate | No |\n"
        "| `lstmd`| Mean + Std | Yes (distributional) |\n"
        "| `hbnn` | Mean + Std | Yes (Bayesian) |\n\n"
        "---\n"
        "Shounak Sushanta Dasgupta (M25AI2015), \nAkarsh Srivastava (M25AI2070), \nRaju Kumar Goswami (M25AI2069), \nSaumyadhi Mukherjee (M25AI2028), \nSarita Rajput (M25AI2076), \nAshutosh Nigam (M25AI2006)"
    ),
    version="1.0.0",
    lifespan=lifespan,
    contact={"name": "Ashutosh Nigam"},
    license_info={"name": "MIT"},
)


class ModelName(str, Enum):
    lstm = "lstm"
    lstmd = "lstmd"
    hbnn = "hbnn"


class Timestep(BaseModel):
    cpu: float = Field(..., ge=0.0, le=1.0, description="CPU utilisation [0, 1]")
    mem: float = Field(..., ge=0.0, le=1.0, description="Memory utilisation [0, 1]")


class PredictRequest(BaseModel):
    model: ModelName = Field(..., description="Model to use for prediction")
    timeseries: List[Timestep] = Field(
        ...,
        description=(
            "Exactly 288 timesteps of CPU and Memory utilisation readings, "
            "ordered from oldest (t-287) to most recent (t). "
            "Each value must be in [0, 1]."
        ),
        min_length=288,
        max_length=288,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "lstm",
                "timeseries": [{"cpu": 0.38, "mem": 0.55}] * 288,
            }
        }
    }


class PredictResponse(BaseModel):
    model: ModelName = Field(..., description="Model used")
    prediction: dict = Field(..., description="Prediction result")
    note: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    type: str
    uncertainty: bool
    description: str
    weight_file: str
    available: bool


@app.get("/", tags=["Health"])
def health_check():
    """Returns API status."""
    return {
        "status": "ok",
        "api": "Workload Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
def list_models():
    """List all three models with their descriptions and weight-file availability."""
    return [
        ModelInfo(
            name="lstm",
            type="LSTM (Long Short-Term Memory)",
            uncertainty=False,
            description=(
                "Deterministic baseline. Outputs a single point estimate "
                "for CPU and Memory. Fast inference, no uncertainty quantification."
            ),
            weight_file=WEIGHT_FILES["lstm"],
            available=os.path.exists(WEIGHT_FILES["lstm"]),
        ),
        ModelInfo(
            name="lstmd",
            type="LSTMD (LSTM with Distributional Output)",
            uncertainty=True,
            description=(
                "LSTM with a probabilistic output layer (IndependentNormal). "
                "Returns mean and standard deviation per resource for "
                "uncertainty-aware resource provisioning."
            ),
            weight_file=WEIGHT_FILES["lstmd"],
            available=os.path.exists(WEIGHT_FILES["lstmd"]),
        ),
        ModelInfo(
            name="hbnn",
            type="HBNN (Heteroscedastic Bayesian Neural Network)",
            uncertainty=True,
            description=(
                "Bayesian neural network with variational inference (DenseVariational). "
                "Captures both epistemic (model) and aleatoric (data) uncertainty."
            ),
            weight_file=WEIGHT_FILES["hbnn"],
            available=os.path.exists(WEIGHT_FILES["hbnn"]),
        ),
    ]


@app.post("/predict", response_model=PredictResponse, tags=["Predict"])
def predict(request: PredictRequest):
    """
    Run a workload prediction using the selected model.

    - **model**: one of `lstm`, `lstmd`, `hbnn`
    - **timeseries**: exactly **288** `{cpu, mem}` objects in chronological order
    | Model | Fields |
    |-------|--------|
    | `lstm` | `cpu`, `mem` — point estimates |
    | `lstmd` | `cpu_mean`, `cpu_std`, `mem_mean`, `mem_std` |
    | `hbnn`  | `cpu_mean`, `cpu_std`, `mem_mean`, `mem_std` |

    All values are normalised utilisation in **[0, 1]**.
    Provision at `mean + 2 × std` for a ~95 % safety margin.
    """
    model_name = request.model.value

    raw = np.array([[ts.cpu, ts.mem] for ts in request.timeseries], dtype=np.float32)
    scaled = _minmax_scale(raw)
    X = scaled[np.newaxis, :, :]

    try:
        model = get_model(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        if model_name == "lstm":
            raw_pred = model.predict(X, verbose=0)
            pred = _minmax_inverse(raw_pred[0])
            prediction = {
                "cpu": float(np.clip(pred[0], 0.0, 1.0)),
                "mem": float(np.clip(pred[1], 0.0, 1.0)),
            }
            note = "Point estimate — no uncertainty quantification."

        else:
            dist = model(X)
            mean_scaled = dist.mean().numpy()[0]
            std_scaled = dist.stddev().numpy()[0]

            mean = _minmax_inverse(mean_scaled)
            std = std_scaled / 2.0

            prediction = {
                "cpu_mean": float(np.clip(mean[0], 0.0, 1.0)),
                "cpu_std": float(np.clip(std[0], 0.0, None)),
                "mem_mean": float(np.clip(mean[1], 0.0, 1.0)),
                "mem_std": float(np.clip(std[1], 0.0, None)),
            }
            note = (
                "Distributional output. Use *_mean for point estimates and "
                "*_std for uncertainty (provision at mean + 2×std for ~95 % safety margin)."
            )

    except Exception as exc:
        logger.exception("Inference error for model '%s'", model_name)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return PredictResponse(model=request.model, prediction=prediction, note=note)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
