import gc
import json
import math
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump
from scipy.stats import zscore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

SEED = 42
LOOKBACK = 168
FORECAST_HORIZON = 24
INITIAL_EPOCHS = 12
INITIAL_PATIENCE = 4
DEFAULT_BATCH_SIZE = 32
MC_DROPOUT_PASSES = 20
ANOMALY_SAMPLE_DAYS = 7
STACKED_LSTM_UNITS = 128
BIDIRECTIONAL_LSTM_UNITS = 80
STACKED_DENSE_UNITS = 64
COMMON_DENSE_UNITS = 48
DATASET_COLUMNS = [
    "datetime",
    "demand_kw",
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
    "dew_point_c",
    "solar_irradiance",
    "is_weekend",
    "is_holiday",
    "hour",
    "day_of_week",
    "month",
    "season",
]
SEASONS = ["winter", "summer", "monsoon", "post_monsoon"]
MODEL_NAMES = ["stacked_lstm", "bidirectional_lstm"]


def set_random_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_tensorflow_runtime() -> list[str]:
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    names = [gpu.name for gpu in gpus]
    print(f"TensorFlow GPUs detected: {names if names else 'none'}")
    return names


def reset_tensorflow_memory() -> None:
    tf.keras.backend.clear_session()
    gc.collect()


def create_output_directories(root: Path) -> dict[str, Path]:
    directories = {
        "root": root,
        "data": root / "data",
        "preprocessing": root / "preprocessing",
        "models": root / "models",
        "training": root / "training",
        "evaluation": root / "evaluation",
        "anomaly": root / "anomaly_testing",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def acquire_dataset(paths: dict[str, Path]) -> tuple[pd.DataFrame, str]:
    data_path = paths["data"] / "electricity_demand_dataset.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {data_path}")
    dataset = pd.read_csv(data_path, parse_dates=["datetime"])
    missing_columns = [column for column in DATASET_COLUMNS if column not in dataset.columns]
    if missing_columns:
        raise RuntimeError(f"Dataset CSV is missing required columns: {missing_columns}")
    return dataset, "Prepared project dataset CSV"


def detect_demand_anomalies(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    iqr_mask = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    z_values = pd.Series(zscore(values.fillna(values.median()), nan_policy="omit"), index=values.index).abs()
    z_mask = z_values > 3.0
    combined = iqr_mask & z_mask
    if combined.sum() == 0:
        combined = iqr_mask
    return combined.fillna(False)


def replace_anomalies_with_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy().sort_values("datetime").reset_index(drop=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.set_index("datetime").ffill().bfill()
    anomalies = detect_demand_anomalies(frame["demand_kw"])
    frame["is_anomaly"] = anomalies.astype(int)
    frame["demand_kw"] = frame["demand_kw"].mask(anomalies).interpolate("time").ffill().bfill()
    return frame.reset_index()


def calculate_heat_index_celsius(temperature_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    temperature_f = temperature_c * 9.0 / 5.0 + 32.0
    humidity = humidity_pct
    heat_index_f = (
        -42.379
        + 2.04901523 * temperature_f
        + 10.14333127 * humidity
        - 0.22475541 * temperature_f * humidity
        - 0.00683783 * temperature_f**2
        - 0.05481717 * humidity**2
        + 0.00122874 * temperature_f**2 * humidity
        + 0.00085282 * temperature_f * humidity**2
        - 0.00000199 * temperature_f**2 * humidity**2
    )
    heat_index_c = (heat_index_f - 32.0) * 5.0 / 9.0
    return pd.Series(np.where(temperature_c >= 26.0, heat_index_c, temperature_c), index=temperature_c.index)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = replace_anomalies_with_interpolation(df)
    frame = frame.sort_values("datetime").reset_index(drop=True)
    frame["lag_24h"] = frame["demand_kw"].shift(24)
    frame["lag_168h"] = frame["demand_kw"].shift(168)
    frame["rolling_12h_mean"] = frame["demand_kw"].rolling(12).mean()
    frame["rolling_24h_mean"] = frame["demand_kw"].rolling(24).mean()
    frame["rolling_168h_mean"] = frame["demand_kw"].rolling(168).mean()
    frame["same_hour_last_week"] = frame["lag_168h"]
    frame["hour_sin"] = np.sin(2.0 * np.pi * frame["hour"] / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * frame["hour"] / 24.0)
    frame["day_of_week_sin"] = np.sin(2.0 * np.pi * frame["day_of_week"] / 7.0)
    frame["day_of_week_cos"] = np.cos(2.0 * np.pi * frame["day_of_week"] / 7.0)
    frame["month_sin"] = np.sin(2.0 * np.pi * frame["month"] / 12.0)
    frame["month_cos"] = np.cos(2.0 * np.pi * frame["month"] / 12.0)
    frame["heat_index"] = calculate_heat_index_celsius(frame["temperature_c"], frame["humidity_pct"])
    for season in SEASONS:
        frame[f"season_{season}"] = (frame["season"] == season).astype(int)
    return frame.dropna().reset_index(drop=True)


def create_lstm_sequences(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    datetimes: pd.Series,
    seasons: pd.Series,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_values = []
    y_values = []
    future_times = []
    future_seasons = []
    dt_values = pd.to_datetime(datetimes).to_numpy()
    season_values = seasons.to_numpy()
    last_start = len(feature_values) - horizon + 1
    for index in range(lookback, last_start):
        x_values.append(feature_values[index - lookback : index])
        y_values.append(target_values[index : index + horizon, 0])
        future_times.append(dt_values[index : index + horizon])
        future_seasons.append(season_values[index : index + horizon])
    return (
        np.asarray(x_values, dtype=np.float32),
        np.asarray(y_values, dtype=np.float32),
        np.asarray(future_times),
        np.asarray(future_seasons),
    )


def preprocess_dataset(df: pd.DataFrame, paths: dict[str, Path]) -> dict[str, object]:
    engineered = engineer_features(df)
    feature_columns = [column for column in engineered.columns if column not in ["datetime", "season"]]
    train_end = int(len(engineered) * 0.70)
    validation_end = int(len(engineered) * 0.85)
    train_df = engineered.iloc[:train_end].copy()
    validation_df = engineered.iloc[train_end:validation_end].copy()
    test_df = engineered.iloc[validation_end:].copy()
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(train_df[feature_columns])
    target_scaler.fit(train_df[["demand_kw"]])
    dump(feature_scaler, paths["preprocessing"] / "feature_scaler.joblib")
    dump(target_scaler, paths["preprocessing"] / "target_scaler.joblib")
    splits: dict[str, dict[str, object]] = {}
    for name, split_df in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
        feature_scaled = feature_scaler.transform(split_df[feature_columns])
        target_scaled = target_scaler.transform(split_df[["demand_kw"]])
        x_values, y_values, times, seasons = create_lstm_sequences(
            feature_scaled,
            target_scaled,
            split_df["datetime"],
            split_df["season"],
            LOOKBACK,
            FORECAST_HORIZON,
        )
        splits[name] = {"X": x_values, "y": y_values, "times": times, "seasons": seasons}
    metadata = {
        "feature_columns": feature_columns,
        "lookback": LOOKBACK,
        "forecast_horizon": FORECAST_HORIZON,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "train_sequences": int(len(splits["train"]["X"])),
        "validation_sequences": int(len(splits["validation"]["X"])),
        "test_sequences": int(len(splits["test"]["X"])),
    }
    (paths["preprocessing"] / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "feature_columns": feature_columns,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "splits": splits,
        "metadata": metadata,
    }


def validate_sequence_availability(preprocessed: dict[str, object]) -> None:
    splits = preprocessed["splits"]
    for name in ["train", "validation", "test"]:
        if len(splits[name]["X"]) == 0:
            raise RuntimeError(f"{name} split has no LSTM sequences; increase data rows or reduce LOOKBACK/FORECAST_HORIZON")


def compile_model(model: Model, learning_rate: float) -> Model:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def build_stacked_lstm(input_shape: tuple[int, int]) -> Model:
    units = STACKED_LSTM_UNITS
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(units, return_sequences=True),
            Dropout(0.2),
            LSTM(max(32, units // 2)),
            Dropout(0.2),
            Dense(STACKED_DENSE_UNITS, activation="relu"),
            Dense(COMMON_DENSE_UNITS, activation="relu"),
            Dense(FORECAST_HORIZON),
        ],
        name="stacked_lstm",
    )
    return compile_model(model, 0.001)


def build_bidirectional_lstm(input_shape: tuple[int, int]) -> Model:
    units = BIDIRECTIONAL_LSTM_UNITS
    model = Sequential(
        [
            Input(shape=input_shape),
            Bidirectional(LSTM(units, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(max(32, units // 2))),
            Dropout(0.3),
            Dense(COMMON_DENSE_UNITS, activation="relu"),
            Dense(FORECAST_HORIZON),
        ],
        name="bidirectional_lstm",
    )
    return compile_model(model, 0.001)


def build_model_by_name(name: str, input_shape: tuple[int, int]) -> Model:
    if name == "stacked_lstm":
        return build_stacked_lstm(input_shape)
    if name == "bidirectional_lstm":
        return build_bidirectional_lstm(input_shape)
    raise ValueError(f"Unsupported model name: {name}")


def train_model(
    model_name: str,
    model: Model,
    preprocessed: dict[str, object],
    paths: dict[str, Path],
) -> tuple[dict[str, list[float]], float]:
    train = preprocessed["splits"]["train"]
    validation = preprocessed["splits"]["validation"]
    callbacks = [EarlyStopping(monitor="val_loss", patience=INITIAL_PATIENCE, restore_best_weights=True)]
    start = time.time()
    history = model.fit(
        train["X"],
        train["y"],
        validation_data=(validation["X"], validation["y"]),
        epochs=INITIAL_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=False,
    )
    elapsed = time.time() - start
    model.save(paths["models"] / f"{model_name}.keras")
    pd.DataFrame(history.history).to_csv(paths["training"] / f"{model_name}_history.csv", index=False)
    return {key: [float(value) for value in values] for key, values in history.history.items()}, elapsed


def train_models(preprocessed: dict[str, object], paths: dict[str, Path]) -> tuple[dict[str, dict[str, list[float]]], dict[str, float]]:
    input_shape = (
        preprocessed["splits"]["train"]["X"].shape[1],
        preprocessed["splits"]["train"]["X"].shape[2],
    )
    histories: dict[str, dict[str, list[float]]] = {}
    times: dict[str, float] = {}
    for model_name in MODEL_NAMES:
        reset_tensorflow_memory()
        model = build_model_by_name(model_name, input_shape)
        history, elapsed = train_model(model_name, model, preprocessed, paths)
        histories[model_name] = history
        times[model_name] = elapsed
        del model
        reset_tensorflow_memory()
    return histories, times


def load_saved_model(model_name: str, paths: dict[str, Path]) -> Model:
    return tf.keras.models.load_model(paths["models"] / f"{model_name}.keras")


def inverse_target_values(values: np.ndarray, target_scaler: MinMaxScaler) -> np.ndarray:
    original_shape = values.shape
    return target_scaler.inverse_transform(values.reshape(-1, 1)).reshape(original_shape)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_flat = actual.reshape(-1)
    predicted_flat = predicted.reshape(-1)
    absolute_error = np.abs(actual_flat - predicted_flat)
    denominator = np.clip(np.abs(actual_flat), 1e-6, None)
    smape_denominator = np.clip((np.abs(actual_flat) + np.abs(predicted_flat)) / 2.0, 1e-6, None)
    return {
        "mae": float(mean_absolute_error(actual_flat, predicted_flat)),
        "rmse": float(np.sqrt(mean_squared_error(actual_flat, predicted_flat))),
        "mape": float(np.mean(absolute_error / denominator) * 100.0),
        "r2": float(r2_score(actual_flat, predicted_flat)),
        "smape": float(np.mean(absolute_error / smape_denominator) * 100.0),
        "max_error": float(np.max(absolute_error)),
        "p95_error": float(np.percentile(absolute_error, 95)),
    }


def save_prediction_intervals(
    model: Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    times: np.ndarray,
    target_scaler: MinMaxScaler,
    output_dir: Path,
) -> None:
    count = min(24 * 7, len(x_test))
    if count == 0:
        return
    sample_x = x_test[:count]
    predictions = []
    for _ in range(MC_DROPOUT_PASSES):
        predictions.append(model(sample_x, training=True).numpy())
    scaled_predictions = np.asarray(predictions)
    mean_scaled = scaled_predictions.mean(axis=0)
    std_scaled = scaled_predictions.std(axis=0)
    mean_kw = inverse_target_values(mean_scaled, target_scaler)
    upper_kw = inverse_target_values(mean_scaled + 1.96 * std_scaled, target_scaler)
    lower_kw = inverse_target_values(mean_scaled - 1.96 * std_scaled, target_scaler)
    actual_kw = inverse_target_values(y_test[:count], target_scaler)
    interval_frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(times[:count, 0]),
            "actual_kw": actual_kw[:, 0],
            "mean_prediction_kw": mean_kw[:, 0],
            "lower_95_kw": lower_kw[:, 0],
            "upper_95_kw": upper_kw[:, 0],
        }
    )
    interval_frame.to_csv(output_dir / "prediction_intervals.csv", index=False)


def save_horizon_degradation(actual: np.ndarray, predicted: np.ndarray, model_name: str, output_dir: Path) -> None:
    rows = []
    for horizon_index in range(actual.shape[1]):
        rows.append(
            {
                "horizon_step": horizon_index + 1,
                "mae": float(mean_absolute_error(actual[:, horizon_index], predicted[:, horizon_index])),
                "rmse": float(np.sqrt(mean_squared_error(actual[:, horizon_index], predicted[:, horizon_index]))),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / f"{model_name}_horizon_degradation.csv", index=False)


def evaluate_models(preprocessed: dict[str, object], paths: dict[str, Path]) -> pd.DataFrame:
    target_scaler = preprocessed["target_scaler"]
    test = preprocessed["splits"]["test"]
    metrics_rows = []
    for model_name in MODEL_NAMES:
        reset_tensorflow_memory()
        model = load_saved_model(model_name, paths)
        predictions_scaled = model.predict(test["X"], batch_size=DEFAULT_BATCH_SIZE, verbose=0)
        predicted = inverse_target_values(predictions_scaled, target_scaler)
        actual = inverse_target_values(test["y"], target_scaler)
        metrics = calculate_metrics(actual, predicted)
        metrics["model"] = model_name
        metrics_rows.append(metrics)
        save_horizon_degradation(actual, predicted, model_name, paths["evaluation"])
        del model
        reset_tensorflow_memory()
    metrics_frame = pd.DataFrame(metrics_rows).sort_values("mae").reset_index(drop=True)
    metrics_frame.to_csv(paths["evaluation"] / "model_comparison.csv", index=False)
    best_model_name = str(metrics_frame.iloc[0]["model"])
    best_model = load_saved_model(best_model_name, paths)
    save_prediction_intervals(
        best_model,
        test["X"],
        test["y"],
        test["times"],
        target_scaler,
        paths["evaluation"],
    )
    del best_model
    reset_tensorflow_memory()
    return metrics_frame


def run_anomaly_injection_test(best_model: Model, preprocessed: dict[str, object], output_dir: Path) -> dict[str, float]:
    test = preprocessed["splits"]["test"]
    target_scaler = preprocessed["target_scaler"]
    feature_columns = preprocessed["feature_columns"]
    sample_count = min(24 * ANOMALY_SAMPLE_DAYS, len(test["X"]))
    if sample_count == 0:
        result = {"clean_mae": math.nan, "injected_mae": math.nan, "mae_degradation": math.nan}
        (output_dir / "anomaly_injection_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    clean_x = test["X"][:sample_count].copy()
    injected_x = clean_x.copy()
    demand_index = feature_columns.index("demand_kw")
    rng = np.random.default_rng(SEED)
    anomaly_positions = rng.choice(sample_count * LOOKBACK, size=min(10, sample_count * LOOKBACK), replace=False)
    for position_index, flat_position in enumerate(anomaly_positions):
        row = flat_position // LOOKBACK
        timestep = flat_position % LOOKBACK
        injected_x[row, timestep, demand_index] = 1.0 if position_index % 2 == 0 else 0.0
    clean_pred = inverse_target_values(best_model.predict(clean_x, verbose=0), target_scaler)
    injected_pred = inverse_target_values(best_model.predict(injected_x, verbose=0), target_scaler)
    actual = inverse_target_values(test["y"][:sample_count], target_scaler)
    clean_metrics = calculate_metrics(actual, clean_pred)
    injected_metrics = calculate_metrics(actual, injected_pred)
    result = {
        "clean_mae": clean_metrics["mae"],
        "injected_mae": injected_metrics["mae"],
        "mae_degradation": injected_metrics["mae"] - clean_metrics["mae"],
        "clean_rmse": clean_metrics["rmse"],
        "injected_rmse": injected_metrics["rmse"],
        "rmse_degradation": injected_metrics["rmse"] - clean_metrics["rmse"],
    }
    (output_dir / "anomaly_injection_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        rows.append([str(row[column]) for column in frame.columns])
    widths = []
    for index, column in enumerate(columns):
        values = [row[index] for row in rows]
        widths.append(max([len(column), *[len(value) for value in values]]))
    header = "| " + " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = ["| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_final_report(
    output_path: Path,
    source: str,
    metrics_frame: pd.DataFrame,
    best_model_name: str,
    anomaly_metrics: dict[str, float],
    total_training_time: float,
) -> None:
    metrics_markdown = dataframe_to_markdown(metrics_frame)
    best_row = metrics_frame.iloc[0]
    text = f"""# Final Report

## Objective
Develop a scalable LSTM-based short-term electricity demand forecasting system for BSES Rajdhani-style operations using temporal demand data, weather variability, anomaly handling, and 24-step forecasting.

## Data Source
{source}

## Models Compared
- stacked_lstm
- bidirectional_lstm

## Best Model
{best_model_name}

## Model Comparison
{metrics_markdown}

## Best Model Metrics
- MAE: {best_row['mae']:.2f}
- RMSE: {best_row['rmse']:.2f}
- MAPE: {best_row['mape']:.2f}%
- R2: {best_row['r2']:.4f}
- sMAPE: {best_row['smape']:.2f}%

## Anomaly Handling Check
- Clean MAE: {anomaly_metrics.get('clean_mae', math.nan):.2f}
- Injected MAE: {anomaly_metrics.get('injected_mae', math.nan):.2f}
- MAE degradation: {anomaly_metrics.get('mae_degradation', math.nan):.2f}

## Training Summary
- Lookback window: {LOOKBACK} hours
- Forecast horizon: {FORECAST_HORIZON} hours
- Total training time: {total_training_time:.2f} seconds
"""
    output_path.write_text(text, encoding="utf-8")


def print_execution_summary(metrics_frame: pd.DataFrame, best_model_name: str, total_training_time: float) -> None:
    print("\nExecution Summary")
    columns = ["model", "mae", "rmse", "mape", "r2", "smape", "max_error", "p95_error"]
    print(metrics_frame[columns].to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Total training time: {total_training_time:.2f} seconds")


def main() -> None:
    start = time.time()
    detected_gpus = configure_tensorflow_runtime()
    set_random_seeds()
    paths = create_output_directories(Path("outputs"))
    dataset, data_source = acquire_dataset(paths)
    preprocessed = preprocess_dataset(dataset, paths)
    validate_sequence_availability(preprocessed)
    _, training_times = train_models(preprocessed, paths)
    metrics_frame = evaluate_models(preprocessed, paths)
    best_model_name = str(metrics_frame.iloc[0]["model"])
    best_model = load_saved_model(best_model_name, paths)
    anomaly_metrics = run_anomaly_injection_test(best_model, preprocessed, paths["anomaly"])
    del best_model
    reset_tensorflow_memory()
    total_training_time = sum(training_times.values())
    write_final_report(
        Path("final_report.md"),
        data_source,
        metrics_frame,
        best_model_name,
        anomaly_metrics,
        total_training_time,
    )
    run_metadata = {
        "best_model_name": best_model_name,
        "data_source": data_source,
        "training_config": {
            "candidate_models": MODEL_NAMES,
            "initial_epochs": INITIAL_EPOCHS,
            "batch_size": DEFAULT_BATCH_SIZE,
            "mc_dropout_passes": MC_DROPOUT_PASSES,
            "anomaly_sample_days": ANOMALY_SAMPLE_DAYS,
        },
        "tensorflow_gpus": detected_gpus,
        "total_training_time_seconds": total_training_time,
        "total_runtime_seconds": time.time() - start,
    }
    (paths["root"] / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print_execution_summary(metrics_frame, best_model_name, total_training_time)


if __name__ == "__main__":
    main()
