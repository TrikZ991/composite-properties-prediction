"""Обучение моделей для приложения app.py.

Повторяет итоговые модели из ноутбука (разделы 6 и 8):
- случайный лес для модуля упругости и прочности при растяжении;
- нейронная сеть MLP для соотношения матрица-наполнитель.

Вместе с моделями сохраняются метрики на тестовой выборке для модели и
для baseline (прогноз средним), чтобы приложение могло честно их показать.

Запуск: python train_models.py
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import callbacks, layers

from src.data import (FEATURES_ML, FEATURES_NN, MODELS_DIR, RANDOM_STATE,
                      TARGET_ELASTICITY, TARGET_RATIO, TARGET_STRENGTH,
                      load_dataset)


def metrics(y_true, y_pred):
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    report = {}

    # Случайный лес: модуль упругости и прочность
    X = df[FEATURES_ML]
    for target, fname in [(TARGET_ELASTICITY, "model_elasticity.joblib"),
                          (TARGET_STRENGTH, "model_strength.joblib")]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, df[target], test_size=0.3, random_state=RANDOM_STATE)
        pipe = Pipeline([("scaler", MinMaxScaler()),
                         ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))])
        grid = {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 10]}
        search = GridSearchCV(pipe, grid, cv=10, scoring="neg_mean_absolute_error", n_jobs=-1)
        model = search.fit(X_tr, y_tr).best_estimator_
        base = DummyRegressor().fit(X_tr, y_tr)
        joblib.dump(model, MODELS_DIR / fname)
        report[target] = {"model": metrics(y_te, model.predict(X_te)),
                          "baseline": metrics(y_te, base.predict(X_te)),
                          "train_mean": float(y_tr.mean()),
                          "model_name": "Случайный лес"}
        print(target, report[target])

    # Нейронная сеть: соотношение матрица-наполнитель
    keras.utils.set_random_seed(RANDOM_STATE)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURES_NN], df[TARGET_RATIO], test_size=0.3, random_state=RANDOM_STATE)
    scaler = MinMaxScaler().fit(X_tr)
    model = keras.Sequential([
        layers.Input(shape=(len(FEATURES_NN),)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mae")
    model.fit(scaler.transform(X_tr), y_tr, epochs=200, batch_size=32, validation_split=0.2, verbose=0,
              callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)])
    pred = model.predict(scaler.transform(X_te), verbose=0).flatten()
    model.save(MODELS_DIR / "nn_model.keras")
    joblib.dump(scaler, MODELS_DIR / "scaler_nn.joblib")
    report[TARGET_RATIO] = {"model": metrics(y_te, pred),
                            "baseline": metrics(y_te, np.full(len(y_te), y_tr.mean())),
                            "train_mean": float(y_tr.mean()),
                            "model_name": "Нейронная сеть (MLP)"}
    print(TARGET_RATIO, report[TARGET_RATIO])

    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Модели и metrics.json сохранены в", MODELS_DIR)


if __name__ == "__main__":
    main()
