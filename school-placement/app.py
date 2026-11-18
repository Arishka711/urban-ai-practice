"""Интерактивное приложение для размещения школ в г. Приморске.

Запуск из корня проекта:
    .\.venv\Scripts\python.exe -m streamlit run school-placement\app.py

Приложение использует проверенный расчётный модуль ``school_model.py`` и
фактический API BlocksNet 1.0.0a9 (``competitive_provision``). Исходные данные
практики № 2 не изменяются: все сценарии считаются на копии вектора
вместимости.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from branca.colormap import LinearColormap
from shapely.geometry import Point
from streamlit_folium import st_folium

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import school_model as sm  # noqa: E402

# ---------------------------------------------------------------------------
# Страница и константы
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Моделирование размещения школ в Приморске",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BLOCK_IDS = list(range(sm.BLOCK_COUNT))
DEFAULT_BLOCK = 16
AUTO_CAPACITY = 600
SCENARIO_A_CAPACITY = 600
SCENARIO_B_CAPACITY = 300
SITE_AREA_WARNING = 1_000.0

MODE_BASELINE = "Исходная обеспеченность"
MODE_SCENARIO = "Обеспеченность после строительства"
MODE_CHANGE = "Изменение обеспеченности"
MAP_MODES = [MODE_BASELINE, MODE_SCENARIO, MODE_CHANGE]
MODE_KEYS = {
    MODE_BASELINE: "baseline",
    MODE_SCENARIO: "scenario",
    MODE_CHANGE: "change",
}

PROVISION_COLORS = ["#b91c1c", "#f97316", "#facc15", "#84cc16", "#15803d"]
CHANGE_COLORS = ["#b91c1c", "#fca5a5", "#f3f4f6", "#86efac", "#15803d"]
AUTO_COLORS = ["#f3f4f6", "#bbf7d0", "#4ade80", "#15803d"]

# ---------------------------------------------------------------------------
# Оформление
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 2.5rem;
            max-width: 1520px;
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetricLabel"] {
            color: #4b5563;
        }
        .app-subtitle {
            color: #4b5563;
            font-size: 1.02rem;
            margin-top: -0.35rem;
            margin-bottom: 1.1rem;
        }
        .section-note {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px 14px;
            color: #334155;
            font-size: 0.93rem;
        }
        .best-pill {
            display: inline-block;
            background: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
            border-radius: 999px;
            padding: 3px 10px;
            margin: 2px 4px 2px 0;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .muted {
            color: #64748b;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Загружаем модель и воспроизводим исходный расчёт…")
def get_model() -> sm.SchoolModel:
    """Одна модель на процесс: она не изменяется сценариями."""
    return sm.load_model()


@st.cache_data(show_spinner=False)
def get_geometry(path_str: str) -> gpd.GeoDataFrame:
    """Геометрия 40 кварталов в WGS84 для Folium."""
    gdf = gpd.read_file(path_str)
    required = {"block_id", "block_area", "geometry"}
    missing = required.difference(gdf.columns)
    if missing:
        raise ValueError(f"В GeoJSON отсутствуют поля: {sorted(missing)}")
    gdf = gdf[["block_id", "block_area", "geometry"]].copy()
    gdf["block_id"] = gdf["block_id"].astype(int)
    gdf["block_area"] = gdf["block_area"].astype(float)
    gdf = gdf.sort_values("block_id").reset_index(drop=True)
    if set(gdf["block_id"]) != set(BLOCK_IDS):
        raise ValueError("Идентификаторы кварталов не совпадают с 0…39.")
    return gdf.to_crs(4326)


# ---------------------------------------------------------------------------
# Состояние и форматирование
# ---------------------------------------------------------------------------


def init_state() -> None:
    defaults: dict[str, Any] = {
        "selected_block": DEFAULT_BLOCK,
        "schools": [],
        "scenario_result": None,
        "auto_result": None,
        "map_mode": MODE_BASELINE,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "cmp_a_block" not in st.session_state:
        # Намеренно не лучший квартал: сценарий А должен быть выбором
        # пользователя, а не автоматической подстановкой оптимума.
        st.session_state["cmp_a_block"] = 4
    if "cmp_b_block_1" not in st.session_state:
        st.session_state["cmp_b_block_1"] = 16
    if "cmp_b_block_2" not in st.session_state:
        st.session_state["cmp_b_block_2"] = 25


def reset_app_state() -> None:
    """Возвращает приложение к исходному состоянию."""
    st.session_state.schools = []
    st.session_state.scenario_result = None
    st.session_state.auto_result = None
    st.session_state.selected_block = DEFAULT_BLOCK
    st.session_state.map_mode = MODE_BASELINE
    for key in (
        "_last_click_manual",
        "_last_click_auto",
        "new_school_capacity",
        "cmp_a_block",
        "cmp_b_block_1",
        "cmp_b_block_2",
    ):
        st.session_state.pop(key, None)


def fmt_int(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    return f"{int(round(float(value))):,}".replace(",", " ")


def fmt_area(value: Any) -> str:
    return f"{fmt_int(value)} м²"


def fmt_pct(value: Any, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "н/д"
    except (TypeError, ValueError):
        return "н/д"
    return f"{float(value) * 100:.{digits}f} %".replace(".", ",")


def fmt_pp(value: Any, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "н/д"
    except (TypeError, ValueError):
        return "н/д"
    number = f"{float(value) * 100:+.{digits}f}".replace(".", ",")
    return f"{number} п.п."


def fmt_signed_int(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return f"{int(round(float(value))):+,}".replace(",", " ")


def block_label(block_id: Any) -> str:
    return f"Квартал {int(block_id)}"


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def scenario_frame(
    baseline: sm.ProvisionResult, scenario: sm.ProvisionResult | None
) -> pd.DataFrame:
    """Единая таблица «до/после» для карты и интерфейса."""
    if scenario is None:
        df = pd.DataFrame(
            {
                "block_id": baseline.block_ids.astype(int),
                "population": baseline.population,
                "demand": baseline.demand,
                "capacity_base": baseline.capacity,
                "capacity_scenario": baseline.capacity,
                "capacity_change": np.zeros(sm.BLOCK_COUNT, dtype=int),
                "deficit_base": baseline.demand_left,
                "deficit_scenario": baseline.demand_left,
                "deficit_change": np.zeros(sm.BLOCK_COUNT, dtype=int),
                "provision_base": baseline.provision_strong,
                "provision_scenario": baseline.provision_strong,
                "provision_change": np.where(
                    np.isnan(baseline.provision_strong), np.nan, 0.0
                ),
            }
        )
    else:
        compared = baseline.compare_with(scenario)
        df = compared.rename(
            columns={
                "capacity_before": "capacity_base",
                "capacity_after": "capacity_scenario",
                "demand_left_before": "deficit_base",
                "demand_left_after": "deficit_scenario",
                "provision_before": "provision_base",
                "provision_after": "provision_scenario",
            }
        )
    return df


def display_block_table(df: pd.DataFrame, has_scenario: bool) -> pd.DataFrame:
    """Русские заголовки и аккуратные строки для st.dataframe."""
    if has_scenario:
        display = df[
            [
                "block_id",
                "population",
                "block_area",
                "capacity_base",
                "capacity_scenario",
                "capacity_change",
                "deficit_base",
                "deficit_scenario",
                "deficit_change",
                "provision_base",
                "provision_scenario",
                "provision_change",
            ]
        ].rename(
            columns={
                "block_id": "Квартал",
                "population": "Население, чел.",
                "block_area": "Площадь, м²",
                "capacity_base": "Вместимость была, мест",
                "capacity_scenario": "Вместимость стала, мест",
                "capacity_change": "Δ вместимости, мест",
                "deficit_base": "Дефицит был, мест",
                "deficit_scenario": "Дефицит стал, мест",
                "deficit_change": "Δ дефицита, мест",
                "provision_base": "Обеспеченность была",
                "provision_scenario": "Обеспеченность стала",
                "provision_change": "Δ обеспеченности, п.п.",
            }
        )
        for column in ("Обеспеченность была", "Обеспеченность стала"):
            display[column] = display[column].map(fmt_pct)
        display["Δ обеспеченности, п.п."] = display[
            "Δ обеспеченности, п.п."
        ].map(fmt_pp)
        display["Δ дефицита, мест"] = display["Δ дефицита, мест"].map(
            fmt_signed_int
        )
        display["Δ вместимости, мест"] = display["Δ вместимости, мест"].map(
            fmt_signed_int
        )
        display["Площадь, м²"] = display["Площадь, м²"].map(fmt_int)
    else:
        display = df[
            [
                "block_id",
                "population",
                "block_area",
                "demand",
                "capacity_base",
                "deficit_base",
                "provision_base",
            ]
        ].rename(
            columns={
                "block_id": "Квартал",
                "population": "Население, чел.",
                "block_area": "Площадь, м²",
                "demand": "Спрос, мест",
                "capacity_base": "Вместимость, мест",
                "deficit_base": "Дефицит, мест",
                "provision_base": "Обеспеченность",
            }
        )
        display["Обеспеченность"] = display["Обеспеченность"].map(fmt_pct)
        display["Площадь, м²"] = display["Площадь, м²"].map(fmt_int)
    return display


def styled_block_table(numeric_df: pd.DataFrame, has_scenario: bool):
    """Таблица по кварталам с подсветкой улучшений в сценарии."""
    display = display_block_table(numeric_df, has_scenario)
    if not has_scenario:
        return display

    prov_col = "Δ обеспеченности, п.п."
    deficit_col = "Δ дефицита, мест"
    styles = pd.DataFrame("", index=display.index, columns=display.columns)
    highlight = "background-color: #dcfce7; color: #166534; font-weight: 600;"

    prov_change = pd.to_numeric(
        numeric_df.get("provision_change"), errors="coerce"
    )
    deficit_change = pd.to_numeric(
        numeric_df.get("deficit_change"), errors="coerce"
    )
    if prov_col in display.columns:
        styles.loc[prov_change > 0, prov_col] = highlight
    if deficit_col in display.columns:
        # Отрицательное Δ дефицита означает сокращение дефицита.
        styles.loc[deficit_change < 0, deficit_col] = highlight

    return display.style.apply(lambda _: styles, axis=None)


def table_for_csv(df: pd.DataFrame, has_scenario: bool) -> pd.DataFrame:
    """Числовая таблица для скачивания (без форматированных строк)."""
    if has_scenario:
        return df[
            [
                "block_id",
                "population",
                "block_area",
                "demand",
                "capacity_base",
                "capacity_scenario",
                "capacity_change",
                "deficit_base",
                "deficit_scenario",
                "deficit_change",
                "provision_base",
                "provision_scenario",
                "provision_change",
            ]
        ].rename(
            columns={
                "block_id": "block_id",
                "population": "population",
                "block_area": "block_area_m2",
                "demand": "demand",
                "capacity_base": "capacity_before",
                "capacity_scenario": "capacity_after",
                "capacity_change": "capacity_change",
                "deficit_base": "demand_left_before",
                "deficit_scenario": "demand_left_after",
                "deficit_change": "deficit_change",
                "provision_base": "provision_before",
                "provision_scenario": "provision_after",
                "provision_change": "provision_change",
            }
        )
    return df[
        [
            "block_id",
            "population",
            "block_area",
            "demand",
            "capacity_base",
            "deficit_base",
            "provision_base",
        ]
    ].rename(
        columns={
            "block_id": "block_id",
            "population": "population",
            "block_area": "block_area_m2",
            "demand": "demand",
            "capacity_base": "capacity",
            "deficit_base": "demand_left",
            "provision_base": "provision_strong",
        }
    )


# ---------------------------------------------------------------------------
# Карта
# ---------------------------------------------------------------------------


def render_map_legend(kind: str, *, vmax: float | None = None) -> None:
    """Показывает цветовую шкалу вне карты, чтобы не перекрывать геометрию."""
    if kind == "provision":
        gradient = ", ".join(PROVISION_COLORS)
        labels = ["0%", "25%", "50%", "75%", "100%"]
        title = "Обеспеченность спроса"
        note = "Доля спроса, покрытая в пределах 15 минут пешей доступности."
    elif kind == "change":
        gradient = ", ".join(CHANGE_COLORS)
        labels = [
            "−100 п.п.<br><span class='muted'>ухудшение</span>",
            "−50 п.п.",
            "0",
            "+50 п.п.",
            "+100 п.п.<br><span class='muted'>улучшение</span>",
        ]
        title = "Изменение обеспеченности"
        note = (
            "Разница между сценарием и исходным состоянием в процентных пунктах: "
            "зелёный — улучшение, красный — ухудшение."
        )
    else:  # auto
        max_value = max(0.0, float(vmax or 0.0))
        gradient = ", ".join(AUTO_COLORS)
        if max_value <= 0:
            tick_labels = ["0", "0", "0"]
        else:
            tick_labels = [
                "0",
                fmt_int(max_value / 2),
                fmt_int(max_value),
            ]
        title = "Сокращение дефицита, мест"
        note = (
            "На сколько мест уменьшается общий дефицит при размещении школы "
            "на 600 мест в квартале."
        )
        st.markdown(
            f"""
            <div class="section-note" style="margin-bottom:0.6rem;">
              <div style="font-weight:600;margin-bottom:6px;">{title}</div>
              <div style="height:14px;border-radius:999px;border:1px solid #d1d5db;
                          background:linear-gradient(to right,{gradient});"></div>
              <div style="display:flex;justify-content:space-between;font-size:12px;
                          color:#374151;margin-top:4px;">
                <span>{tick_labels[0]}</span><span>{tick_labels[1]}</span><span>{tick_labels[2]}</span>
              </div>
              <div class="muted" style="margin-top:4px;">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="section-note" style="margin-bottom:0.6rem;">
          <div style="font-weight:600;margin-bottom:6px;">{title}</div>
          <div style="height:14px;border-radius:999px;border:1px solid #d1d5db;
                      background:linear-gradient(to right,{gradient});"></div>
          <div style="display:flex;justify-content:space-between;font-size:12px;
                      color:#374151;margin-top:4px;gap:8px;">
            <span>{labels[0]}</span><span>{labels[1]}</span><span>{labels[2]}</span>
            <span>{labels[3]}</span><span style="text-align:right;">{labels[4]}</span>
          </div>
          <div class="muted" style="margin-top:4px;">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _value_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def _add_school_marker(
    folium_map: folium.Map,
    row: pd.Series,
    total_capacity: int,
    label: str,
) -> None:
    point = row.geometry.representative_point()
    html = (
        '<div style="font-size:22px;line-height:22px;text-align:center;'
        'filter:drop-shadow(0 1px 2px rgba(15,23,42,.35));">🏫</div>'
    )
    folium.Marker(
        location=[point.y, point.x],
        tooltip=folium.Tooltip(f"{label}: {fmt_int(total_capacity)} мест", sticky=True),
        popup=folium.Popup(
            f"<b>{label}</b><br>Квартал {int(row.block_id)}<br>"
            f"Новая вместимость: {fmt_int(total_capacity)} мест",
            max_width=280,
        ),
        icon=folium.DivIcon(
            html=html,
            icon_size=(28, 28),
            icon_anchor=(14, 14),
        ),
    ).add_to(folium_map)


def make_map(
    geo: gpd.GeoDataFrame,
    data: pd.DataFrame,
    *,
    mode: str = "baseline",
    selected_block: int | None = None,
    schools: list[dict[str, Any]] | None = None,
    auto: bool = False,
    best_blocks: list[int] | None = None,
) -> folium.Map:
    """Строит карту кварталов; значения берутся из той же таблицы, что и в UI."""
    map_gdf = geo.merge(data, on="block_id", how="left", suffixes=("", "_data"))
    best_set = {int(x) for x in (best_blocks or [])}
    schools = schools or []

    if auto:
        value_column = "deficit_reduction"
        map_gdf["value"] = map_gdf[value_column].map(_value_or_none)
        map_gdf["tt_capacity"] = map_gdf["total_capacity"].map(fmt_int)
        map_gdf["tt_provision"] = map_gdf["aggregate_provision"].map(fmt_pct)
        map_gdf["tt_deficit"] = map_gdf["total_demand_left"].map(fmt_int)
        map_gdf["tt_extra"] = map_gdf["deficit_reduction"].map(fmt_int)
        provision_alias = "Обеспеченность после"
        extra_alias = "Сокращение дефицита, мест"
        cmap = LinearColormap(
            AUTO_COLORS,
            vmin=0.0,
            vmax=max(1.0, float(map_gdf["value"].max(skipna=True) or 1.0)),
            caption="Сокращение дефицита, мест",
        )
    else:
        if mode == "baseline":
            value_column = "provision_base"
            capacity_column = "capacity_base"
            deficit_column = "deficit_base"
        else:
            value_column = (
                "provision_scenario" if mode == "scenario" else "provision_change"
            )
            capacity_column = "capacity_scenario"
            deficit_column = "deficit_scenario"

        map_gdf["value"] = map_gdf[value_column].map(_value_or_none)
        map_gdf["tt_capacity"] = map_gdf[capacity_column].map(fmt_int)
        map_gdf["tt_deficit"] = map_gdf[deficit_column].map(fmt_int)
        if mode == "baseline":
            map_gdf["tt_provision"] = map_gdf["provision_base"].map(fmt_pct)
            provision_alias = "Обеспеченность"
            map_gdf["tt_extra"] = "—"
        else:
            map_gdf["tt_provision"] = map_gdf["provision_scenario"].map(fmt_pct)
            provision_alias = "Обеспеченность после"
            map_gdf["tt_extra"] = map_gdf["provision_change"].map(fmt_pp)
        extra_alias = "Δ обеспеченности"
        if mode == "change":
            cmap = LinearColormap(
                CHANGE_COLORS,
                vmin=-1.0,
                vmax=1.0,
                caption="Изменение обеспеченности",
            )
        else:
            cmap = LinearColormap(
                PROVISION_COLORS,
                vmin=0.0,
                vmax=1.0,
                caption="Обеспеченность (доля спроса)",
            )

    map_gdf["tt_population"] = map_gdf["population"].map(fmt_int)
    map_gdf["tt_area"] = map_gdf["block_area"].map(fmt_int)
    map_gdf["is_best"] = map_gdf["block_id"].isin(best_set)
    map_gdf["tt_status"] = "—"
    if auto:
        map_gdf.loc[map_gdf["is_best"], "tt_status"] = "равноценный оптимальный"
    if selected_block is not None:
        map_gdf.loc[
            map_gdf["block_id"].eq(int(selected_block)), "tt_status"
        ] = "выбранный квартал"
        if map_gdf.loc[
            map_gdf["block_id"].eq(int(selected_block)), "is_best"
        ].any():
            map_gdf.loc[
                map_gdf["block_id"].eq(int(selected_block)), "tt_status"
            ] = "выбранный; равноценный оптимальный"

    center = [
        float((map_gdf.geometry.bounds["miny"].min() + map_gdf.geometry.bounds["maxy"].max()) / 2),
        float((map_gdf.geometry.bounds["minx"].min() + map_gdf.geometry.bounds["maxx"].max()) / 2),
    ]
    folium_map = folium.Map(
        location=center,
        zoom_start=13,
        tiles=None,
        control_scale=True,
    )
    # Стандартная подложка OpenStreetMap без ключа. Если тайлы недоступны,
    # остаётся нейтральный фон, на котором видны кварталы и маркеры школ.
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        attr="&copy; OpenStreetMap contributors",
        control=False,
    ).add_to(folium_map)
    folium_map.get_root().html.add_child(
        folium.Element(
            "<style>.leaflet-container{background:#e5e7eb;}</style>"
        )
    )

    def style_function(feature: dict[str, Any]) -> dict[str, Any]:
        properties = feature.get("properties", {})
        block_id = int(properties.get("block_id", -1))
        value = properties.get("value")
        if value is None:
            fill_color = "#d1d5db"
        else:
            fill_color = cmap(float(value))
        is_selected = selected_block is not None and block_id == int(selected_block)
        is_best = bool(properties.get("is_best"))
        if is_selected:
            border_color, border_weight = "#111827", 4
        elif is_best:
            border_color, border_weight = "#15803d", 3
        else:
            border_color, border_weight = "#6b7280", 1
        return {
            "fillColor": fill_color,
            "color": border_color,
            "weight": border_weight,
            "fillOpacity": 0.78 if (is_selected or is_best) else 0.62,
            "opacity": 0.95,
        }

    def highlight_function(feature: dict[str, Any]) -> dict[str, Any]:
        return {
            "weight": 4,
            "color": "#111827",
            "fillOpacity": 0.82,
        }

    geo_data = json.loads(map_gdf.to_json())
    folium.GeoJson(
        geo_data,
        name="Кварталы",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "block_id",
                "tt_population",
                "tt_area",
                "tt_capacity",
                "tt_provision",
                "tt_deficit",
                "tt_extra",
                "tt_status",
            ],
            aliases=[
                "Квартал",
                "Население, чел.",
                "Площадь, м²",
                "Вместимость, мест",
                provision_alias,
                "Дефицит, мест",
                extra_alias,
                "Статус",
            ],
            localize=True,
            sticky=True,
            labels=True,
        ),
    ).add_to(folium_map)

    # Маркеры проектируемых школ: по одному на квартал, вместимости суммируются.
    capacities_by_block: dict[int, list[int]] = {}
    labels_by_block: dict[int, str] = {}
    for school in schools:
        block_id = int(school["block_id"])
        capacities_by_block.setdefault(block_id, []).append(int(school["capacity"]))
        labels_by_block.setdefault(block_id, str(school.get("label", "Проектируемая школа")))
    for block_id, capacities in capacities_by_block.items():
        rows = map_gdf.loc[map_gdf["block_id"].eq(block_id)]
        if rows.empty:
            continue
        total = sum(capacities)
        label = labels_by_block[block_id]
        if len(capacities) > 1:
            label = f"{label} ({len(capacities)} школы)"
        _add_school_marker(folium_map, rows.iloc[0], total, label)

    if selected_block is not None:
        rows = map_gdf.loc[map_gdf["block_id"].eq(int(selected_block))]
        if not rows.empty:
            point = rows.iloc[0].geometry.representative_point()
            folium.CircleMarker(
                location=[point.y, point.x],
                radius=14,
                color="#111827",
                weight=2,
                fill=False,
                dash_array="5,5",
                tooltip=f"Выбранный квартал {int(selected_block)}",
            ).add_to(folium_map)

    # Цветовая шкала выносится в интерфейс Streamlit над/под картой,
    # чтобы не перекрывать геометрию кварталов.

    if auto:
        legend_extra = (
            '<span style="display:inline-block;width:18px;height:0;'
            'border-top:4px solid #15803d;vertical-align:middle;"></span> '
            "равноценный оптимальный квартал<br>"
        )
    else:
        legend_extra = (
            '<span style="display:inline-block;width:18px;height:0;'
            'border-top:4px solid #111827;vertical-align:middle;"></span> '
            "выбранный квартал<br>"
        )
    legend_html = (
        '<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; '
        'background: rgba(255,255,255,.95); border: 1px solid #d1d5db; '
        'border-radius: 10px; padding: 9px 12px; font-size: 12px; '
        'color: #111827; box-shadow: 0 4px 12px rgba(15,23,42,.10);">'
        "<b>Условные обозначения</b><br>"
        '<span style="font-size:16px">🏫</span> проектируемая школа<br>'
        f"{legend_extra}"
        "Наведите курсор на квартал для подробностей"
        "</div>"
    )
    folium_map.get_root().html.add_child(folium.Element(legend_html))
    return folium_map


def clicked_block(
    map_state: dict[str, Any] | None,
    geo: gpd.GeoDataFrame,
    prefix: str,
) -> int | None:
    """Определяет квартал по клику на карту; защищает от повторного клика."""
    if not map_state:
        return None
    click = map_state.get("last_clicked")
    if not click:
        return None
    lat = click.get("lat")
    lng = click.get("lng")
    if lat is None or lng is None:
        return None

    key = (round(float(lat), 7), round(float(lng), 7))
    state_key = f"_last_click_{prefix}"
    if st.session_state.get(state_key) == key:
        return None
    st.session_state[state_key] = key

    point = Point(float(lng), float(lat))
    hits = geo.loc[geo.geometry.contains(point)]
    if not hits.empty:
        return int(hits.iloc[0]["block_id"])

    distances = geo.geometry.distance(point)
    nearest_index = distances.idxmin()
    # Допуск около 150 м: помогает при клике точно на границу квартала.
    if float(distances.loc[nearest_index]) <= 0.0015:
        return int(geo.loc[nearest_index, "block_id"])
    return None


def manual_block_selectbox(current: int) -> int:
    """Выпадающий список, который синхронизируется с кликом по карте."""
    if int(current) not in BLOCK_IDS:
        current = DEFAULT_BLOCK
    # Уникальный ключ на выбранный квартал: после клика по карте виджет
    # переинициализируется с правильным index.
    widget_key = f"manual_block_{int(current)}"
    choice = st.selectbox(
        "Квартал",
        BLOCK_IDS,
        index=BLOCK_IDS.index(int(current)),
        format_func=block_label,
        key=widget_key,
        help="Можно выбрать здесь или нажать на квартал на карте.",
    )
    if int(choice) != int(current):
        st.session_state.selected_block = int(choice)
        st.rerun()
    return int(current)


# ---------------------------------------------------------------------------
# Автоматический поиск
# ---------------------------------------------------------------------------


def run_auto_search(model: sm.SchoolModel) -> sm.AutoSearchResult | None:
    """Запускает перебор с индикатором выполнения; ошибки не скрываются."""
    progress = st.progress(0.0, text="Проверено 0 из 40 кварталов")
    status = st.empty()

    def callback(done: int, total: int, block_id: int) -> None:
        progress.progress(
            min(1.0, done / total),
            text=f"Проверено {done} из {total} кварталов",
        )
        status.caption(f"Последний проверенный квартал: {block_id}")

    try:
        auto = model.auto_search_one_school(
            AUTO_CAPACITY,
            progress_callback=callback,
        )
    except Exception as exc:  # ошибка расчёта не подменяется нулём
        progress.empty()
        status.empty()
        st.error(f"Автоматический подбор не выполнен: {exc}")
        return None

    progress.progress(
        1.0,
        text=f"Готово: проверено {auto.checked_count} из {auto.attempted_count}",
    )
    status.success("Перебор завершён.")
    return auto


def representative_best(
    auto: sm.AutoSearchResult, model: sm.SchoolModel
) -> int:
    """Один из равноценных лучших кварталов: самый населённый, затем меньший ID."""
    population = model.population
    return max(
        auto.best_block_ids,
        key=lambda bid: (int(population[bid]), -int(bid)),
    )


# ---------------------------------------------------------------------------
# Вкладка «Ручное размещение»
# ---------------------------------------------------------------------------


def render_manual_tab(
    model: sm.SchoolModel,
    geo: gpd.GeoDataFrame,
    baseline: sm.ProvisionResult,
) -> None:
    scenario = st.session_state.scenario_result
    map_df = scenario_frame(baseline, scenario).merge(
        geo[["block_id", "block_area"]], on="block_id", how="left"
    )
    selected = int(st.session_state.selected_block)
    if selected not in BLOCK_IDS:
        selected = DEFAULT_BLOCK
        st.session_state.selected_block = selected

    st.subheader("Ручное размещение школ")
    st.write(
        "Выберите квартал, укажите вместимость и добавьте одну или несколько "
        "школ. Сценарий всегда начинается от исходной вместимости 250 мест."
    )

    map_column, control_column = st.columns([1.62, 1.0], gap="large")

    with control_column:
        st.markdown("#### Параметры новой школы")
        selected = manual_block_selectbox(selected)
        capacity = st.number_input(
            "Вместимость школы, мест",
            min_value=1,
            max_value=10_000,
            value=600,
            step=1,
            format="%d",
            key="new_school_capacity",
            help="Только целое положительное число мест.",
        )

        add_col, second_col = st.columns(2)
        if add_col.button(
            "➕ Добавить школу",
            type="primary",
            width="stretch",
            help="Добавьте первую, затем при необходимости вторую школу.",
        ):
            try:
                valid_capacity = sm.validate_capacity(capacity)
                schools = list(st.session_state.schools)
                schools.append(
                    {"block_id": int(selected), "capacity": int(valid_capacity)}
                )
                st.session_state.schools = schools
                st.session_state.scenario_result = None
                st.toast(f"Школа добавлена в квартал {selected}.")
            except sm.SchoolModelError as exc:
                st.error(str(exc))
        if second_col.button(
            "Добавить ещё одну",
            width="stretch",
            help="Тот же квартал и та же вместимость будут добавлены второй школой.",
        ):
            try:
                valid_capacity = sm.validate_capacity(capacity)
                schools = list(st.session_state.schools)
                schools.append(
                    {"block_id": int(selected), "capacity": int(valid_capacity)}
                )
                st.session_state.schools = schools
                st.session_state.scenario_result = None
                st.toast("Вторая школа добавлена.")
            except sm.SchoolModelError as exc:
                st.error(str(exc))

        schools = list(st.session_state.schools)
        st.markdown("#### Школы в сценарии")
        if not schools:
            st.info("Школы пока не добавлены. Показано исходное состояние.")
        else:
            for index, school in enumerate(schools):
                school_col, capacity_col, delete_col = st.columns([2.1, 1.3, 0.8])
                school_col.markdown(
                    f"**Школа {index + 1}**  \n{block_label(school['block_id'])}"
                )
                capacity_col.markdown(
                    f"**{fmt_int(school['capacity'])}**  \nмест"
                )
                if delete_col.button(
                    "Удалить",
                    key=f"delete_school_{index}",
                    width="stretch",
                    help="Удалить только эту школу.",
                ):
                    updated = list(st.session_state.schools)
                    updated.pop(index)
                    st.session_state.schools = updated
                    st.session_state.scenario_result = None
                    st.rerun()

            new_total = sum(int(s["capacity"]) for s in schools)
            st.caption(
                f"Новых мест: {fmt_int(new_total)}. "
                f"Итоговая вместимость: {fmt_int(baseline.total_capacity + new_total)}."
            )

        calculate_col, reset_col = st.columns(2)
        if calculate_col.button(
            "Рассчитать сценарий",
            type="primary",
            width="stretch",
            disabled=not schools,
        ):
            try:
                placements = [
                    (int(s["block_id"]), int(s["capacity"])) for s in schools
                ]
                st.session_state.scenario_result = model.place_schools(placements)
                # После успешного расчёта сразу показываем карту «после».
                st.session_state.map_mode = MODE_SCENARIO
                st.toast("Сценарий рассчитан.")
            except sm.SchoolModelError as exc:
                st.session_state.scenario_result = None
                st.error(f"Расчёт не выполнен: {exc}")
        if reset_col.button(
            "Сбросить всё",
            width="stretch",
            help="Очищает сценарий, автоподбор и возвращает исходное состояние.",
        ):
            reset_app_state()
            st.rerun()

        if st.session_state.schools and st.session_state.scenario_result is None:
            st.warning(
                "Школы добавлены, но сценарий ещё не рассчитан. Нажмите "
                "«Рассчитать сценарий», чтобы пересчитать обеспеченность."
            )

        scenario = st.session_state.scenario_result
        map_df = scenario_frame(baseline, scenario).merge(
            geo[["block_id", "block_area"]], on="block_id", how="left"
        )
        selected_row = map_df.loc[map_df["block_id"].eq(selected)].iloc[0]

        st.markdown("---")
        st.markdown(f"#### Выбранный квартал {selected}")
        if selected_row["block_area"] < SITE_AREA_WARNING:
            st.warning(
                "Очень малый по площади квартал: размещение школы требует "
                "отдельного градостроительного обоснования."
            )
        metric_a, metric_b = st.columns(2)
        metric_a.metric(
            "Население",
            f"{fmt_int(selected_row['population'])} чел.",
            help="Модельная численность населения квартала.",
        )
        metric_b.metric(
            "Площадь",
            fmt_area(selected_row["block_area"]),
        )
        metric_c, metric_d = st.columns(2)
        if scenario is None:
            metric_c.metric(
                "Вместимость",
                f"{fmt_int(selected_row['capacity_base'])} мест",
            )
            metric_d.metric(
                "Обеспеченность",
                fmt_pct(selected_row["provision_base"]),
                help="Для кварталов без населения обеспеченность не определена.",
            )
            st.metric(
                "Дефицит",
                f"{fmt_int(selected_row['deficit_base'])} мест",
            )
        else:
            metric_c.metric(
                "Вместимость",
                f"{fmt_int(selected_row['capacity_scenario'])} мест",
                delta=f"{int(selected_row['capacity_change']):+d} мест",
                delta_color="normal",
            )
            metric_d.metric(
                "Обеспеченность",
                fmt_pct(selected_row["provision_scenario"]),
                delta=fmt_pp(selected_row["provision_change"]),
                delta_color="normal",
            )
            st.metric(
                "Дефицит",
                f"{fmt_int(selected_row['deficit_scenario'])} мест",
                delta=f"{fmt_signed_int(selected_row['deficit_change'])} мест",
                delta_color="inverse",
            )

    with map_column:
        mode = st.radio(
            "Режим отображения",
            MAP_MODES,
            key="map_mode",
            horizontal=True,
        )
        mode_key = MODE_KEYS[mode]
        if scenario is None and mode_key != "baseline":
            st.info(
                "Режимы «после строительства» и «изменение» доступны после "
                "расчёта сценария. Сейчас показана исходная обеспеченность."
            )
            mode_key = "baseline"

        render_map_legend("change" if mode_key == "change" else "provision")

        schools_for_map = [
            {
                "block_id": int(school["block_id"]),
                "capacity": int(school["capacity"]),
                "label": "Проектируемая школа",
            }
            for school in st.session_state.schools
        ]
        folium_map = make_map(
            geo,
            map_df,
            mode=mode_key,
            selected_block=selected,
            schools=schools_for_map,
        )
        map_state = st_folium(
            folium_map,
            key="manual_map",
            height=565,
            use_container_width=True,
            returned_objects=["last_clicked"],
        )
        st.caption(
            "🏫 — проектируемая школа; толстая рамка — выбранный квартал. "
            "Нажмите на квартал, чтобы выбрать его. Для подложки OpenStreetMap "
            "нужен интернет; при недоступности тайлов кварталы и школы остаются "
            "видимыми на нейтральном фоне."
        )

        clicked = clicked_block(map_state, geo, "manual")
        if clicked is not None and clicked != int(st.session_state.selected_block):
            st.session_state.selected_block = int(clicked)
            st.rerun()

    st.markdown("---")
    st.subheader("Итоги сценария")
    if scenario is None:
        st.caption("Показано исходное состояние. Добавьте школы и нажмите «Рассчитать сценарий».")
        metric_cols = st.columns(6)
        metric_cols[0].metric("Население", f"{fmt_int(baseline.total_population)} чел.")
        metric_cols[1].metric("Потребность", f"{fmt_int(baseline.total_demand)} мест")
        metric_cols[2].metric("Вместимость", f"{fmt_int(baseline.total_capacity)} мест")
        metric_cols[3].metric("Дефицит", f"{fmt_int(baseline.total_demand_left)} мест")
        metric_cols[4].metric("Обеспеченность", fmt_pct(baseline.aggregate_provision))
        metric_cols[5].metric("Новых школ", "0")
    else:
        deficit_before = baseline.total_demand_left
        deficit_after = scenario.total_demand_left
        reduction = deficit_before - deficit_after
        provision_before = baseline.aggregate_provision
        provision_after = scenario.aggregate_provision
        metric_cols = st.columns(6)
        metric_cols[0].metric("Дефицит был", f"{fmt_int(deficit_before)} мест")
        metric_cols[1].metric(
            "Дефицит стал",
            f"{fmt_int(deficit_after)} мест",
            delta=f"{deficit_after - deficit_before:+d} мест",
            delta_color="inverse",
        )
        metric_cols[2].metric("Сокращение дефицита", f"{fmt_int(reduction)} мест")
        metric_cols[3].metric("Обеспеченность была", fmt_pct(provision_before))
        metric_cols[4].metric(
            "Обеспеченность стала",
            fmt_pct(provision_after),
            delta=fmt_pp(provision_after - provision_before),
        )
        metric_cols[5].metric(
            "Новых школ / мест",
            f"{len(st.session_state.schools)} / "
            f"{fmt_int(sum(int(s['capacity']) for s in st.session_state.schools))}",
        )

    st.subheader("Показатели по кварталам")
    display = styled_block_table(map_df, has_scenario=scenario is not None)
    st.dataframe(display, width="stretch", hide_index=True, height=430)
    st.caption(
        "Дефицит считается по demand_left; обеспеченность — по provision_strong. "
        "Для кварталов без населения обеспеченность не определена (н/д), дефицит равен 0. "
        "Отрицательное Δ дефицита означает улучшение. При наличии сценария зелёным "
        "выделены положительное изменение обеспеченности и сокращение дефицита."
    )

    csv_table = table_for_csv(map_df, has_scenario=scenario is not None)
    file_suffix = "scenario" if scenario is not None else "baseline"
    st.download_button(
        "⬇️ Скачать показатели по кварталам (CSV)",
        data=csv_bytes(csv_table),
        file_name=f"primorsk_schools_{file_suffix}.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Как читать показатели"):
        st.markdown(
            """
            - **Спрос** — расчётная потребность в местах, `ceil(население / 1000 × 120)`
              по каждому кварталу. Спрос не меняется при строительстве школы.
            - **Дефицит** — `demand_left`, то есть потребность, которая не покрыта
              в пределах 15 минут пешей доступности.
            - **Обеспеченность квартала** — `provision_strong = demand_within / demand`.
              Она не усредняется по кварталам; обеспеченность города считается как
              отношение сумм.
            - **Δ обеспеченности** в таблице указана в процентных пунктах.
            """
        )


# ---------------------------------------------------------------------------
# Вкладка «Автоматический подбор»
# ---------------------------------------------------------------------------


def render_auto_tab(model: sm.SchoolModel, geo: gpd.GeoDataFrame) -> None:
    st.subheader("Автоматический подбор")
    st.write(
        "Полный перебор всех 40 кварталов для одной новой школы на 600 мест. "
        "Каждый вариант рассчитывается заново от исходной вместимости 250 мест."
    )

    if st.button(
        "🔎 Найти оптимальное размещение школы на 600 мест",
        type="primary",
        width="stretch",
    ):
        auto = run_auto_search(model)
        if auto is not None:
            st.session_state.auto_result = auto
            st.rerun()

    auto = st.session_state.auto_result
    if auto is None:
        st.info(
            "Нажмите кнопку выше. Будут проверены все кварталы, включая "
            "кварталы без населения и очень малые кварталы."
        )
        return

    if not auto.complete:
        st.warning(
            f"Перебор неполный: проверено {auto.checked_count} из "
            f"{auto.attempted_count}. Результат нельзя называть лучшим среди всех."
        )
    else:
        st.success(
            f"Полный перебор завершён: проверено {auto.checked_count} из "
            f"{auto.attempted_count} кварталов."
        )

    if auto.errors:
        with st.expander("Ошибки отдельных вариантов", expanded=True):
            st.dataframe(
                pd.DataFrame(auto.errors),
                width="stretch",
                hide_index=True,
            )

    best = auto.ranking.loc[auto.ranking["is_best"]]
    best_row = best.iloc[0] if not best.empty else auto.ranking.iloc[0]
    metric_cols = st.columns(5)
    metric_cols[0].metric(
        "Проверено кварталов",
        f"{auto.checked_count} из {auto.attempted_count}",
    )
    metric_cols[1].metric(
        "Лучший остаточный дефицит",
        f"{fmt_int(auto.best_total_deficit)} мест",
    )
    metric_cols[2].metric(
        "Сокращение дефицита",
        f"{fmt_int(auto.best_reduction)} мест",
    )
    metric_cols[3].metric(
        "Итоговая обеспеченность",
        fmt_pct(float(best_row["aggregate_provision"])),
    )
    metric_cols[4].metric(
        "Равноценных вариантов",
        fmt_int(len(auto.best_block_ids)),
    )

    st.markdown(
        "**Равноценные оптимальные кварталы:** "
        + " ".join(
            f'<span class="best-pill">{block_label(x)}</span>'
            for x in auto.best_block_ids
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Все перечисленные варианты дают одинаковый остаточный дефицит "
        f"{fmt_int(auto.best_total_deficit)} мест. Они не скрываются и не "
        "заменяются одним «лучшим» кварталом."
    )

    population_by_block = model.baseline.to_frame()[["block_id", "population"]]
    auto_map_data = (
        auto.ranking.merge(
            geo[["block_id", "block_area"]], on="block_id", how="left"
        )
        .merge(population_by_block, on="block_id", how="left")
    )
    best_schools = [
        {
            "block_id": int(block_id),
            "capacity": int(auto.capacity),
            "label": "Оптимальный вариант",
        }
        for block_id in auto.best_block_ids
    ]
    auto_vmax = float(auto_map_data["deficit_reduction"].max(skipna=True) or 0.0)
    render_map_legend("auto", vmax=auto_vmax)
    folium_map = make_map(
        geo,
        auto_map_data,
        auto=True,
        best_blocks=auto.best_block_ids,
        schools=best_schools,
    )
    st_folium(
        folium_map,
        key="auto_map",
        height=560,
        use_container_width=True,
        returned_objects=[],
    )
    st.caption(
        "Цвет квартала — сокращение дефицита при размещении в нём школы на "
        "600 мест. Зелёная рамка и значки 🏫 — равноценные оптимальные кварталы. "
        "Для подложки OpenStreetMap нужен интернет; при недоступности тайлов "
        "кварталы и школы остаются видимыми на нейтральном фоне."
    )

    st.subheader("Все проверенные варианты")
    ranking = auto_map_data.sort_values("rank").copy()
    ranking_display = ranking[
        [
            "rank",
            "block_id",
            "block_area",
            "new_capacity_in_block",
            "total_capacity",
            "total_demand_left",
            "deficit_reduction",
            "aggregate_provision",
            "aggregate_provision_change",
            "is_best",
        ]
    ].rename(
        columns={
            "rank": "Ранг",
            "block_id": "Квартал",
            "block_area": "Площадь, м²",
            "new_capacity_in_block": "Вместимость в квартале, мест",
            "total_capacity": "Всего вместимости, мест",
            "total_demand_left": "Остаточный дефицит, мест",
            "deficit_reduction": "Сокращение дефицита, мест",
            "aggregate_provision": "Обеспеченность",
            "aggregate_provision_change": "Δ обеспеченности",
            "is_best": "Оптимальный",
        }
    )
    for column in ("Обеспеченность",):
        ranking_display[column] = ranking_display[column].map(fmt_pct)
    ranking_display["Δ обеспеченности"] = ranking_display[
        "Δ обеспеченности"
    ].map(fmt_pp)
    ranking_display["Площадь, м²"] = ranking_display["Площадь, м²"].map(fmt_int)
    ranking_display["Оптимальный"] = ranking_display["Оптимальный"].map(
        lambda value: "да" if bool(value) else "нет"
    )

    ranking_styles = pd.DataFrame(
        "", index=ranking_display.index, columns=ranking_display.columns
    )
    highlight = "background-color: #dcfce7; color: #166534; font-weight: 600;"
    ranking_styles.loc[
        pd.to_numeric(ranking["deficit_reduction"], errors="coerce") > 0,
        "Сокращение дефицита, мест",
    ] = highlight
    ranking_styles.loc[
        pd.to_numeric(ranking["aggregate_provision_change"], errors="coerce") > 0,
        "Δ обеспеченности",
    ] = highlight
    styled_ranking = ranking_display.style.apply(lambda _: ranking_styles, axis=None)

    st.dataframe(
        styled_ranking,
        width="stretch",
        hide_index=True,
        height=560,
    )
    st.caption(
        "Ранг 1 присвоен всем вариантам с максимальным сокращением дефицита. "
        "Сортировка: сокращение дефицита, затем остаточный дефицит, затем ID квартала."
    )

    ranking_csv = ranking[
        [
            "rank",
            "block_id",
            "block_area",
            "new_capacity_in_block",
            "total_capacity",
            "total_demand",
            "total_demand_left",
            "deficit_reduction",
            "aggregate_provision",
            "aggregate_provision_change",
            "is_best",
        ]
    ].rename(
        columns={
            "rank": "rank",
            "block_id": "block_id",
            "block_area": "block_area_m2",
            "new_capacity_in_block": "capacity_in_block_after",
            "total_capacity": "total_capacity",
            "total_demand": "total_demand",
            "total_demand_left": "total_demand_left",
            "deficit_reduction": "deficit_reduction",
            "aggregate_provision": "aggregate_provision",
            "aggregate_provision_change": "aggregate_provision_change",
            "is_best": "is_best",
        }
    )
    st.download_button(
        "⬇️ Скачать рейтинг автоматического перебора (CSV)",
        data=csv_bytes(ranking_csv),
        file_name=f"primorsk_auto_search_{auto.capacity}.csv",
        mime="text/csv",
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Вкладка «Сравнение»
# ---------------------------------------------------------------------------


def comparison_rows(
    baseline: sm.ProvisionResult,
    result_a: sm.ProvisionResult,
    result_b: sm.ProvisionResult,
    result_c: sm.ProvisionResult | None,
    *,
    a_block: int,
    b_block_1: int,
    b_block_2: int,
    c_block: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        name: str,
        blocks: str,
        new_schools: str,
        result: sm.ProvisionResult,
    ) -> None:
        rows.append(
            {
                "Сценарий": name,
                "Кварталы": blocks,
                "Новые школы": new_schools,
                "Вместимость, мест": result.total_capacity,
                "Остаточный дефицит, мест": result.total_demand_left,
                "Сокращение дефицита, мест": (
                    baseline.total_demand_left - result.total_demand_left
                ),
                "Обеспеченность, %": result.aggregate_provision * 100,
                "Δ обеспеченности, п.п.": (
                    result.aggregate_provision - baseline.aggregate_provision
                )
                * 100,
            }
        )

    add("Исходное состояние", "—", "—", baseline)
    add(
        "А: одна школа 600",
        block_label(a_block),
        "1 × 600",
        result_a,
    )
    b_blocks = (
        block_label(b_block_1)
        if int(b_block_1) == int(b_block_2)
        else f"{block_label(b_block_1)} + {block_label(b_block_2)}"
    )
    add("Б: две школы по 300", b_blocks, "2 × 300", result_b)
    if result_c is not None and c_block is not None:
        add(
            "В: автоматический выбор 600",
            f"{block_label(c_block)} (один из лучших)",
            "1 × 600",
            result_c,
        )
    return pd.DataFrame(rows)


def render_comparison_tab(
    model: sm.SchoolModel,
    baseline: sm.ProvisionResult,
) -> None:
    st.subheader("Сравнение сценариев")
    st.write(
        "Сценарии А и Б используют выбранные вами кварталы. Приложение не "
        "подменяет их оптимальными вариантами. Сценарий В — результат полного "
        "автоматического перебора."
    )

    select_a, select_b1, select_b2 = st.columns(3)
    with select_a:
        a_block = st.selectbox(
            "Сценарий А: квартал",
            BLOCK_IDS,
            format_func=block_label,
            key="cmp_a_block",
        )
    with select_b1:
        b_block_1 = st.selectbox(
            "Сценарий Б: квартал 1",
            BLOCK_IDS,
            format_func=block_label,
            key="cmp_b_block_1",
        )
    with select_b2:
        b_block_2 = st.selectbox(
            "Сценарий Б: квартал 2",
            BLOCK_IDS,
            format_func=block_label,
            key="cmp_b_block_2",
        )
    st.caption(
        "А: одна школа на 600 мест. Б: две школы по 300 мест; если кварталы "
        "совпадают, вместимости суммируются в одном квартале."
    )

    try:
        result_a = model.place_one_school(int(a_block), SCENARIO_A_CAPACITY)
        result_b = model.place_two_schools(
            int(b_block_1),
            SCENARIO_B_CAPACITY,
            int(b_block_2),
            SCENARIO_B_CAPACITY,
        )
    except sm.SchoolModelError as exc:
        st.error(f"Не удалось рассчитать сценарии А и Б: {exc}")
        return

    auto = st.session_state.auto_result
    if auto is None:
        st.info(
            "Для сценария В нужен автоматический перебор. Его можно выполнить "
            "здесь, не переключаясь на другую вкладку."
        )
        if st.button(
            "Выполнить автоматический подбор для сценария В",
            type="primary",
            width="stretch",
        ):
            auto_result = run_auto_search(model)
            if auto_result is not None:
                st.session_state.auto_result = auto_result
                st.rerun()
        result_c = None
        c_block = None
        c_best_blocks: list[int] = []
    else:
        if not auto.complete:
            st.warning(
                f"Автоматический перебор неполный: проверено {auto.checked_count} "
                f"из {auto.attempted_count} кварталов. Сценарий В нельзя "
                "интерпретировать как лучший среди всех."
            )
        c_best_blocks = list(auto.best_block_ids)
        c_block = representative_best(auto, model)
        result_c = auto.scenario_results.get(int(c_block))
        if result_c is None:
            st.warning(
                "Для выбранного автоматического варианта нет сохранённого "
                "результата расчёта. Выполните автоматический подбор заново."
            )

    comparison = comparison_rows(
        baseline,
        result_a,
        result_b,
        result_c,
        a_block=int(a_block),
        b_block_1=int(b_block_1),
        b_block_2=int(b_block_2),
        c_block=c_block,
    )

    st.subheader("Сравнительная таблица")
    display = comparison.copy()
    for column in ("Вместимость, мест", "Остаточный дефицит, мест", "Сокращение дефицита, мест"):
        display[column] = display[column].map(fmt_int)
    display["Обеспеченность, %"] = display["Обеспеченность, %"].map(
        lambda value: fmt_pct(float(value) / 100.0)
    )
    display["Δ обеспеченности, п.п."] = display["Δ обеспеченности, п.п."].map(
        lambda value: f"{float(value):+.2f}".replace(".", ",") + " п.п."
    )
    st.dataframe(display, width="stretch", hide_index=True)

    if result_c is not None:
        st.caption(
            "Сценарий В представлен одним из равноценных лучших кварталов. "
            "Равноценные варианты: "
            + ", ".join(block_label(x) for x in c_best_blocks)
            + ". Все они дают одинаковый суммарный дефицит."
        )

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        fig_deficit = px.bar(
            comparison,
            x="Сценарий",
            y="Остаточный дефицит, мест",
            color="Сценарий",
            text_auto=True,
            title="Остаточный дефицит, мест",
            template="plotly_white",
            color_discrete_sequence=["#94a3b8", "#60a5fa", "#a78bfa", "#34d399"],
        )
        fig_deficit.update_layout(
            showlegend=False,
            height=390,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title=None,
        )
        st.plotly_chart(fig_deficit, width="stretch")
    with chart_col_2:
        fig_provision = px.bar(
            comparison,
            x="Сценарий",
            y="Обеспеченность, %",
            color="Сценарий",
            text_auto=".2f",
            title="Обеспеченность территории, %",
            template="plotly_white",
            color_discrete_sequence=["#94a3b8", "#60a5fa", "#a78bfa", "#34d399"],
        )
        fig_provision.update_layout(
            showlegend=False,
            height=390,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title=None,
            yaxis_range=[0, 100],
        )
        st.plotly_chart(fig_provision, width="stretch")

    st.subheader("Краткий вывод")
    best_deficit = float(comparison["Остаточный дефицит, мест"].min())
    leaders = comparison.loc[
        comparison["Остаточный дефицит, мест"].eq(best_deficit), "Сценарий"
    ].tolist()
    if len(leaders) == len(comparison):
        conclusion = (
            "В рамках выбранных кварталов все сценарии дают одинаковый "
            f"остаточный дефицит: {fmt_int(best_deficit)} мест."
        )
    elif len(leaders) == 1:
        conclusion = (
            f"Наименьший остаточный дефицит — {fmt_int(best_deficit)} мест — "
            f"у сценария «{leaders[0]}»."
        )
    else:
        conclusion = (
            f"Одинаковый наименьший остаточный дефицит — {fmt_int(best_deficit)} "
            "мест — у сценариев: "
            + ", ".join(f"«{name}»" for name in leaders)
            + "."
        )
    st.info(
        conclusion
        + " Это сравнение вариантов в рамках принятой модели, а не выбор "
        "готового проекта: возможность строительства и стоимость не оценивались."
    )

    csv_comparison = comparison.copy()
    st.download_button(
        "⬇️ Скачать сравнение сценариев (CSV)",
        data=csv_bytes(csv_comparison),
        file_name="primorsk_school_scenarios_comparison.csv",
        mime="text/csv",
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Вкладка «Методика и ограничения»
# ---------------------------------------------------------------------------


def render_method_tab(model: sm.SchoolModel, baseline: sm.ProvisionResult) -> None:
    st.subheader("Методика и ограничения")
    st.write(
        "Приложение не заменяет расчёт обеспеченности упрощённой формулой. "
        "Оно вызывает фактический метод BlocksNet `competitive_provision` на "
        "сохранённых данных практики № 2."
    )

    st.markdown("### Что считается")
    st.markdown(
        f"""
        - Анализируются **40 кварталов** Приморска с идентификаторами 0…39.
        - Модельное население: **{fmt_int(baseline.total_population)} чел.**
        - Исходная потребность: **{fmt_int(baseline.total_demand)} мест**.
        - Исходная вместимость: **{fmt_int(baseline.total_capacity)} мест**,
          вся она находится в квартале 21.
        - Исходный остаточный дефицит: **{fmt_int(baseline.total_demand_left)} мест**.
        - Исходная агрегированная обеспеченность:
          **{fmt_pct(baseline.aggregate_provision)}**.
        """
    )

    st.markdown("### Параметры BlocksNet")
    st.markdown(
        """
        | Параметр | Значение |
        |---|---|
        | Версия BlocksNet | 1.0.0a9 |
        | Метод | `competitive_provision` |
        | Порог доступности | 15 минут пешей доступности |
        | Норматив спроса | 120 мест на 1000 жителей |
        | `self_supply` | `True` |
        | `max_depth` | `1` |
        | Матрица доступности | 40×40, пешеходная, 5 км/ч |
        | Решатель | целочисленная LP-задача BlocksNet (PuLP/CBC) |
        """
    )
    st.markdown(
        """
        Спрос рассчитывается по каждому кварталу как
        `ceil(население / 1000 × 120)`. Поэтому суммарный спрос равен 1148,
        а не 1130,04: это эффект округления вверх по каждому кварталу.
        При строительстве школы спрос не пересчитывается, меняется только
        вектор вместимости.
        """
    )

    st.markdown("### Как считается сценарий")
    st.markdown(
        """
        - Базовая вместимость `capacity_base` берётся из сохранённых данных и
          не изменяется.
        - Для каждого сценария создаётся копия вектора вместимости; в выбранном
          квартале к ней прибавляется вместимость новой школы.
        - Если две школы находятся в одном квартале, их вместимости суммируются.
        - Весь город пересчитывается заново: `demand_left`, `demand_within`,
          `provision_strong` и агрегированная обеспеченность.
        - Дефицит территории — сумма `demand_left`, а не `capacity_within`.
          Сумма `capacity_within` равна 249 из-за особенностей учёта
          собственной школы квартала 21.
        - Обеспеченность города — `sum(demand_within) / sum(demand)`, а не
          среднее по кварталам.
        """
    )

    st.markdown("### Автоматический подбор")
    st.markdown(
        f"""
        Перебираются все 40 кварталов, включая шесть кварталов без населения
        (11, 17, 18, 19, 30, 37) и очень малые кварталы. Для каждого варианта
        школа на {AUTO_CAPACITY} мест добавляется к исходной вместимости, после
        чего выполняется полный расчёт BlocksNet. Варианты ранжируются по
        сокращению дефицита; все варианты с одинаковым лучшим результатом
        показываются как равноценные. Ошибки отдельных вариантов фиксируются,
        а не заменяются нулевыми значениями.
        """
    )

    st.markdown("### Ограничения")
    st.markdown(
        """
        1. **Население 9417 чел. — модельное.** Оно получено в практике № 2 из
           зданий с допущениями о этажности, жилой площади и нормах, а не из
           официальной статистики.
        2. **Вместимость существующей школы 250 мест — модельная.** В OSM у
           школы нет поля вместимости; 250 — минимальное значение справочника
           BlocksNet `impute_services`.
        3. **В OSM только одна школа.** Сеть, вероятно, неполная; исходные
           21,78% обеспеченности и дефицит 898 мест условны.
        4. **Только пешая доступность.** Матрица построена для пешеходов со
           скоростью 5 км/ч; общественный транспорт и автомобиль не учитываются.
        5. **Расчёт по кварталам, а не по участкам.** Выбор конкретного
           земельного участка, проверка площади, стоимости и нормативных
           ограничений требуют отдельного градостроительного обоснования.
        6. **Шесть кварталов без населения.** Для них `provision_strong = NaN`;
           дефицит равен 0, обеспеченность не определена. Это не ошибка расчёта.
        7. **Очень малые кварталы.** Кварталы 11, 17 и 30 имеют площадь менее
           400 м²; размещение школы на 600 мест в них физически нереалистично,
           хотя формально они участвуют в переборе.
        8. **Результаты относительны.** Сравнение сценариев действительно только
           в рамках принятой модели и сохранённых данных.
        """
    )

    st.markdown("### Источники данных")
    st.markdown(
        """
        - `data/primorsk_blocksnet/blocks_all_metrics.geojson` — геометрия,
          население, вместимость, расчётный спрос и исходная обеспеченность.
        - `data/primorsk_blocksnet/accessibility_matrix_blocks_minutes.csv` —
          матрица пешеходной доступности 40×40.
        - `school-placement/school_model.py` — проверенный расчётный модуль.
        - `school-placement/data_audit.md` — аудит данных и допущений.
        """
    )
    st.caption(
        "Исходные файлы практики № 2 не изменяются. Все сценарии считаются "
        "на копиях вектора вместимости; население и матрица доступности "
        "остаются неизменными."
    )


# ---------------------------------------------------------------------------
# Основная страница
# ---------------------------------------------------------------------------


def main() -> None:
    model = get_model()
    baseline = model.baseline
    geo = get_geometry(str(sm.DEFAULT_DATA_DIR / "blocks_all_metrics.geojson"))
    init_state()

    st.title("🏫 Моделирование размещения школ в городе Приморске")
    st.markdown(
        '<div class="app-subtitle">Практическая работа № 3 · BlocksNet 1.0.0a9 · '
        "40 кварталов · пешеходная доступность 15 минут</div>",
        unsafe_allow_html=True,
    )

    top_metrics = st.columns(5)
    top_metrics[0].metric(
        "Модельное население",
        f"{fmt_int(baseline.total_population)} чел.",
        help="Расчётная численность населения из практики № 2, не официальная статистика.",
    )
    top_metrics[1].metric(
        "Исходная потребность",
        f"{fmt_int(baseline.total_demand)} мест",
        help="Сумма ceil(население / 1000 × 120) по 40 кварталам.",
    )
    top_metrics[2].metric(
        "Исходная вместимость",
        f"{fmt_int(baseline.total_capacity)} мест",
        help="Модельная вместимость одной существующей школы в квартале 21.",
    )
    top_metrics[3].metric(
        "Исходный дефицит",
        f"{fmt_int(baseline.total_demand_left)} мест",
        help="Сумма demand_left: потребность, не покрытая в пределах 15 минут.",
    )
    top_metrics[4].metric(
        "Исходная обеспеченность",
        fmt_pct(baseline.aggregate_provision),
        help="sum(demand_within) / sum(demand), а не среднее по кварталам.",
    )

    st.warning(
        "Население 9 417 чел. и вместимость существующей школы 250 мест — "
        "**модельные величины**. В OSM у школы нет поля вместимости; 250 — "
        "минимальное значение справочника BlocksNet. Это не официальные данные."
    )

    tab_manual, tab_auto, tab_compare, tab_method = st.tabs(
        [
            "Размещение школ",
            "Автоматический подбор",
            "Сравнение",
            "Методика и ограничения",
        ]
    )

    with tab_manual:
        render_manual_tab(model, geo, baseline)
    with tab_auto:
        render_auto_tab(model, geo)
    with tab_compare:
        render_comparison_tab(model, baseline)
    with tab_method:
        render_method_tab(model, baseline)


if __name__ == "__main__":
    main()
