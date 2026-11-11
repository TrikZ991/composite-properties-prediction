"""Загрузка и подготовка данных о композиционных материалах.

Общий код для ноутбука, скрипта обучения и приложения, чтобы данные
везде готовились одинаково.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42

TARGET_ELASTICITY = "Модуль упругости при растяжении, ГПа"
TARGET_STRENGTH = "Прочность при растяжении, МПа"
TARGET_RATIO = "Соотношение матрица-наполнитель"
TARGETS = [TARGET_ELASTICITY, TARGET_STRENGTH]

FEATURES_ML = [
    "Соотношение матрица-наполнитель", "Плотность, кг/м3",
    "модуль упругости, ГПа", "Количество отвердителя, м.%",
    "Содержание эпоксидных групп,%_2", "Температура вспышки, С_2",
    "Поверхностная плотность, г/м2", "Потребление смолы, г/м2",
    "Угол нашивки, град", "Шаг нашивки", "Плотность нашивки",
]
FEATURES_NN = [c for c in FEATURES_ML if c != TARGET_RATIO]


def load_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Объединяет два исходных файла по `id` (INNER JOIN)."""
    bp = pd.read_excel(data_dir / "X_bp.xlsx")
    nup = pd.read_excel(data_dir / "X_nup.xlsx")
    return bp.merge(nup, on="id", how="inner")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка и кодирование, как обосновано в ноутбуке.

    - удаляется `id`;
    - удаляется запись с нулевой плотностью нашивки (ошибка регистрации);
    - угол нашивки 0/90 кодируется как 0/1.
    """
    df = df.drop(columns="id")
    df = df[df["Плотность нашивки"] > 0].reset_index(drop=True)
    df["Угол нашивки, град"] = df["Угол нашивки, град"].replace(90, 1)
    return df


def load_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return prepare(load_raw(data_dir))
