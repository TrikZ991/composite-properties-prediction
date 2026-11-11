"""Streamlit-приложение: прогноз свойств композиционных материалов.

Показывает прогноз моделей рядом с baseline (средним значением), потому что
исследование показало: на этих данных модели не превосходят прогноз средним.

Запуск: streamlit run app.py
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras

from src.data import (FEATURES_ML, FEATURES_NN, MODELS_DIR, TARGET_ELASTICITY,
                      TARGET_RATIO, TARGET_STRENGTH)


@st.cache_resource
def load_models():
    return (joblib.load(MODELS_DIR / "model_elasticity.joblib"),
            joblib.load(MODELS_DIR / "model_strength.joblib"),
            keras.models.load_model(MODELS_DIR / "nn_model.keras"),
            joblib.load(MODELS_DIR / "scaler_nn.joblib"),
            json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8")))


model_elasticity, model_strength, model_nn, scaler_nn, metrics = load_models()

st.title("Прогнозирование свойств композиционных материалов")
st.write("Введите свойства компонентов композита: приложение спрогнозирует конечные "
         "свойства материала и рекомендует соотношение матрица-наполнитель.")

with st.sidebar:
    st.header("Параметры композита")
    ratio = st.number_input("Соотношение матрица-наполнитель", 0.5, 10.0, 2.0, 0.01)
    density = st.number_input("Плотность, кг/м3", 1500.0, 2200.0, 2000.0, 1.0)
    elasticity_comp = st.number_input("Модуль упругости, ГПа", 100.0, 1000.0, 740.0, 0.1)
    hardener = st.number_input("Количество отвердителя, м.%", 20.0, 150.0, 100.0, 0.1)
    epoxy = st.number_input("Содержание эпоксидных групп, %", 15.0, 35.0, 22.0, 0.1)
    flash_temp = st.number_input("Температура вспышки, °C", 90.0, 420.0, 285.0, 0.1)
    surface_density = st.number_input("Поверхностная плотность, г/м2", 50.0, 1500.0, 210.0, 1.0)
    resin = st.number_input("Потребление смолы, г/м2", 50.0, 800.0, 220.0, 1.0)
    angle = st.selectbox("Угол нашивки, град", [0, 90])
    step = st.number_input("Шаг нашивки", 1.0, 20.0, 5.0, 0.5)
    patch_density = st.number_input("Плотность нашивки", 20.0, 120.0, 55.0, 1.0)

row = pd.DataFrame([{
    "Соотношение матрица-наполнитель": ratio,
    "Плотность, кг/м3": density,
    "модуль упругости, ГПа": elasticity_comp,
    "Количество отвердителя, м.%": hardener,
    "Содержание эпоксидных групп,%_2": epoxy,
    "Температура вспышки, С_2": flash_temp,
    "Поверхностная плотность, г/м2": surface_density,
    "Потребление смолы, г/м2": resin,
    "Угол нашивки, град": 1 if angle == 90 else 0,  # бинарное кодирование
    "Шаг нашивки": step,
    "Плотность нашивки": patch_density,
}])

if st.button("Рассчитать прогноз"):
    preds = {
        TARGET_ELASTICITY: model_elasticity.predict(row[FEATURES_ML])[0],
        TARGET_STRENGTH: model_strength.predict(row[FEATURES_ML])[0],
        TARGET_RATIO: model_nn.predict(scaler_nn.transform(row[FEATURES_NN]), verbose=0)[0][0],
    }
    labels = {TARGET_ELASTICITY: ("Модуль упругости при растяжении", "ГПа", "{:.1f}"),
              TARGET_STRENGTH: ("Прочность при растяжении", "МПа", "{:.0f}"),
              TARGET_RATIO: ("Рекомендуемое соотношение матрица-наполнитель", "", "{:.2f}")}

    st.subheader("Результаты прогноза")
    for col, (target, pred) in zip(st.columns(3), preds.items()):
        title, unit, fmt = labels[target]
        mean = metrics[target]["train_mean"]
        col.metric(title, f"{fmt.format(pred)} {unit}".strip(),
                   delta=f"{fmt.format(pred - mean)} от среднего", delta_color="off")

    st.subheader("Качество моделей на тестовой выборке")
    table = pd.DataFrame([{
        "Прогноз": labels[t][0],
        "Модель": m["model_name"],
        "MAE модели": round(m["model"]["mae"], 3),
        "MAE baseline (среднее)": round(m["baseline"]["mae"], 3),
        "R² модели": round(m["model"]["r2"], 3),
    } for t, m in metrics.items()])
    st.dataframe(table, hide_index=True)
    st.caption("R² ≈ 0 означает, что модель предсказывает не лучше, чем среднее значение по выборке.")
