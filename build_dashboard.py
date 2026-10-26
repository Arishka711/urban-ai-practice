#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_dashboard.py
Генерация standalone dashboard.html по результатам практической №2 (BlocksNet)
для города Приморск (Выборгский район, Ленинградская область).

Скрипт ТОЛЬКО читает уже рассчитанные GeoJSON и не повторяет расчёты.
Дашборд открывается в обычном браузере (двойной клик по dashboard.html).

Основные источники (только чтение):
    data/primorsk_blocksnet/blocks_all_metrics.geojson      - сводка по 40 кварталам (53 поля)
    data/primorsk_blocksnet/blocks_spacematrix.geojson     - морфотипы Spacematrix (столбец morphotype)
    data/primorsk_blocksnet/blocks_strelka.geojson         - классификация «Стрелки» (столбец morphotype)

Результат:
    dashboard.html  - единый файл дашборда в корне проекта
"""

import html
import io
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

# ----------------------------------------------------------------------------- #
# 0. Пути и константы
# ----------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "primorsk_blocksnet"
GEOJSON_MAIN = DATA_DIR / "blocks_all_metrics.geojson"
GEOJSON_SPACE = DATA_DIR / "blocks_spacematrix.geojson"
GEOJSON_STRELKA = DATA_DIR / "blocks_strelka.geojson"
OUT_HTML = ROOT / "dashboard.html"

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"

# Человеческие подписи столбцов (фактические имена полей из blocks_all_metrics)
COLUMN_LABELS = {
    "block_id": "№ квартала",
    "land_use_name": "Землепользование (доминирующее)",
    "share_RESIDENTIAL": "Доля RESIDENTIAL, доли ед.",
    "share_BUSINESS": "Доля BUSINESS, доли ед.",
    "share_INDUSTRIAL": "Доля INDUSTRIAL, доли ед.",
    "share_SPECIAL": "Доля SPECIAL, доли ед.",
    "share_TRANSPORT": "Доля TRANSPORT, доли ед.",
    "share_AGRICULTURE": "Доля AGRICULTURE, доли ед.",
    "share_covered": "Покрыто зонами, доли ед.",
    "share_no_data": "Не покрыто (no data), доли ед.",
    "block_area": "Площадь квартала, м²",
    "build_footprint_in_block": "Площадь пятна застройки, м²",
    "population_model": "Модели населения, чел.",
    "schools_capacity": "Вместимость школы (восст.), мест",
    "demand": "Спрос на школы, чел.",
    "capacity": "Вместимость (восст.), мест",
    "demand_left": "Необеспеченный спрос, чел.",
    "demand_within": "Спрос в пределах пешей доступности (within), чел.",
    "demand_without": "Спрос вне пешей доступности (without), чел.",
    "capacity_left": "Остаток вместимости, мест",
    "capacity_within": "Вместимость в пределах пешей доступности (within), мест",
    "capacity_without": "Вместимость вне пешей доступности (without), мест",
    "provision_strong": "provision_strong (доли ед.)",
    "provision_weak": "provision_weak (доли ед.)",
    "footprint_sum": "Площадь пятна застройки, м²",
    "bfloor_sum": "Поэтажная площадь, м²",
    "build_ratio": "GSI (доли ед.)",
    "fsi": "FSI / FAR (доли ед.)",
    "site_area": "Площадь территории, м²",
    "footprint_area": "Площадь пятна (obs.), м²",
    "build_floor_area": "Поэтажная площадь (obs.), м²",
    "cluster_spacematrix": "Кластер Spacematrix",
    "l_spacematrix": "l (FSI/GSI, усреднённая этажность)",
    "mxi_spacematrix": "mxi (индекс морфологической смеси)",
    "morphotype_spacematrix": "Морфотип Spacematrix",
    "morphotype_strelka": "Морфотип «Стрелки»",
    "area_accessibility": "Area accessibility, мин (взвеш. ср. время пешком)",
}


def fmt(x, nd=2):
    """Форматирование числа для вывода в HTML и tooltips."""
    if x is None or (isinstance(x, float) and (pd.isna(x) or np.isinf(x))):
        return "нет данных"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(xf - round(xf)) < 1e-9 and abs(xf) >= 1000:
        return f"{int(round(xf)):,}".replace(",", " ")
    return f"{xf:.{nd}f}"


def fmt_int(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "нет данных"
    return f"{int(round(float(x))):,}".replace(",", " ")


def fmt_share(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "нет данных"
    return f"{float(x) * 100:.1f} %"


# ----------------------------------------------------------------------------- #
# 1. Загрузка сохранённых данных (БЕЗ пересчёта)
# ----------------------------------------------------------------------------- #
def load_data():
    gdf = gpd.read_file(GEOJSON_MAIN)
    if gdf.crs is None:
        gdf = gdf.set_crs(32635)
    gdf_wgs = gdf.to_crs(4326).copy()

    # Отдельные файлы с одноимённым столбцом morphotype - НЕ смешиваем
    sm = gpd.read_file(GEOJSON_SPACE)
    st = gpd.read_file(GEOJSON_STRELKA)
    sm_type = sm.set_index("block_id")["morphotype"]
    st_type = st.set_index("block_id")["morphotype"]

    gdf_wgs["morphotype_spacematrix"] = gdf_wgs["block_id"].map(sm_type)
    gdf_wgs["morphotype_strelka"] = gdf_wgs["block_id"].map(st_type)
    return gdf, gdf_wgs, sm, st


# ----------------------------------------------------------------------------- #
# 2. Готовим таблицу по кварталам (псевдо-агрегированные строки для таблицы)
# ----------------------------------------------------------------------------- #
def prepare_table(gdf_wgs: gpd.GeoDataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=gdf_wgs.index)
    df["№ квартала"] = gdf_wgs["block_id"].astype(int).astype(str)
    df["Землепользование"] = gdf_wgs["land_use_name"].fillna("нет данных")
    df["Площадь квартала, м²"] = gdf_wgs["block_area"].round(0)
    df["Доля RESIDENTIAL, %"] = (gdf_wgs["share_RESIDENTIAL"] * 100).round(1)
    df["Доля BUSINESS, %"] = (gdf_wgs["share_BUSINESS"] * 100).round(1)
    df["GSI (доли ед.)"] = gdf_wgs["build_ratio"].round(4)
    df["FSI (доли ед.)"] = gdf_wgs["fsi"].round(4)
    df["Модели населения, чел."] = gdf_wgs["population_model"].fillna(0).astype(int)
    df["Спрос школы, чел."] = gdf_wgs["demand"].astype(int)
    df["provision_strong, %"] = (gdf_wgs["provision_strong"].fillna(np.nan) * 100).round(1)
    df["provision_weak, %"] = (gdf_wgs["provision_weak"].fillna(np.nan) * 100).round(1)
    df["Spacematrix"] = gdf_wgs["morphotype_spacematrix"].fillna("нет данных")
    df["«Стрелки»"] = gdf_wgs["morphotype_strelka"].fillna("нет данных")
    df["Пешеходная доступность, мин"] = gdf_wgs["area_accessibility"].round(1)
    return df


# ----------------------------------------------------------------------------- #
# 3. Карта Folium со слоями всех показателей
# ----------------------------------------------------------------------------- #
LANDUSE_COLORS = {
    "RESIDENTIAL": "#e05c4b",
    "BUSINESS": "#f2a54a",
    "AGRICULTURE": "#63a86c",
    "SPECIAL": "#9b59b6",
    "INDUSTRIAL": "#7f8c8d",
    "TRANSPORT": "#5d6d7e",
}

LEGEND_COLORS = {
    "RESIDENTIAL (жилое)": "#e05c4b",
    "BUSINESS (деловое)": "#f2a54a",
    "AGRICULTURE (сель/хоз)": "#63a86c",
    "Недостаточно данных": "#b6bcbf",
}


def _add_geojson_layer(fmap, gdf_wgs, fill_hex, opacity, tooltip_list, name, show=True):
    """Вспомогательная функция: добавить GeoJSON слой с общим стилем."""
    docs = []
    for geom in gdf_wgs.geometry:
        feature_json = {
            "type": "Feature",
            "properties": {},
            "geometry": geom.__geo_interface__,
        }
        docs.append(feature_json)
    style_fn = f"function(feature) {{ return {{ fillColor:'{fill_hex}', color:'#3b4252', weight:1, fillOpacity:{opacity}; }}; }}"
    # folium питает style_function в Python, поэтovamente пишем напрямую.
    layer = folium.GeoJson(
        data={"type": "FeatureCollection", "features": docs},
        name=name,
        show=show,
        style_function=lambda feature: {
            "fillColor": fill_hex,
            "color": "#3b4252",
            "weight": 1,
            "fillOpacity": opacity,
        },
    )
    folium.GeoJsonPopup(
        names=tooltip_list, locales=False, labels=False, style="font-size:12px;"
    )
    return layer


def build_map(gdf_wgs: gpd.GeoDataFrame) -> str:
    """Собирает Folium карту с несколькими переключаемыми слоями (легенды - в HTML)."""
    # Центр — в проекции, затем перевод в WGS84 для Folium
    center_pt = gdf_wgs.to_crs(32635).geometry.centroid.union_all()
    center_wgs = gpd.GeoSeries([center_pt], crs=32635).to_crs(4326).iloc[0]
    if hasattr(center_wgs, "y"):
        cx, cy = center_wgs.x, center_wgs.y
    else:  # MultiPoint
        pts = list(center_wgs.geoms)
        cx = sum(p.x for p in pts) / len(pts)
        cy = sum(p.y for p in pts) / len(pts)
    fmap = folium.Map(
        location=[cy, cx],
        zoom_start=13,
        tiles="CartoDB positron",
    )

    # --- подготовка вспомогательных функций для слоёв ------------------------- #
    def style(color, opacity):
        return lambda _feat: {
            "fillColor": color,
            "color": "#3b4252",
            "weight": 1,
            "fillOpacity": opacity,
        }

    def add_layer(col_fmt, name, show=False, color_fn=None, tooltip_fn=None):
        """Добавляет FeatureGroup с полигонами и tooltips."""
        fg = folium.FeatureGroup(name=name, show=show)
        for _, row in gdf_wgs.iterrows():
            if row.geometry is None:
                continue
            props = {
                "row": row,
                "__html": tooltip_fn(row) if tooltip_fn else "",
            }
            color = color_fn(row) if color_fn else "#b6bcbf"
            gj = folium.GeoJson(
                data={"type": "Feature", "properties": {}, "geometry": row.geometry.__geo_interface__},
                style_function=style(color, 0.75),
            )
            gj.add_child(folium.Tooltip(tooltip_fn(row) if tooltip_fn else ""))
            fg.add_child(gj)
        fmap.add_child(fg)

    # ---- Слой 1: землепользование ------------------------------------------- #
    def landuse_color(row):
        lu = row.get("land_use_name")
        if lu is None or (isinstance(lu, float) and pd.isna(lu)):
            return "#b6bcbf"
        return LANDUSE_COLORS.get(lu, "#b6bcbf")

    def landuse_tooltip(row):
        lu = row.get("land_use_name")
        lu_txt = "<нет данных>" if lu is None or (isinstance(lu, float) and pd.isna(lu)) else str(lu)
        parts = [
            f"<b>Квартал №{int(row['block_id'])}</b>",
            f"Землепользование: {lu_txt}",
            f"Покрыто зонами: {fmt_share(row.get('share_covered'))}",
            f"Не покрыто (no data): {fmt_share(row.get('share_no_data'))}",
        ]
        for k, label in [
            ("share_RESIDENTIAL", "RESIDENTIAL"),
            ("share_BUSINESS", "BUSINESS"),
            ("share_INDUSTRIAL", "INDUSTRIAL"),
            ("share_SPECIAL", "SPECIAL"),
            ("share_TRANSPORT", "TRANSPORT"),
            ("share_AGRICULTURE", "AGRICULTURE"),
        ]:
            if row.get(k) and row.get(k) > 0:
                parts.append(f"{label}: {fmt_share(row.get(k))}")
        return "<br>".join(parts)

    landuse_layer = folium.FeatureGroup(name="Землепользование", show=True)
    for _, row in gdf_wgs.iterrows():
        gj = folium.GeoJson(
            data={"type": "Feature", "properties": {}, "geometry": row.geometry.__geo_interface__},
            style_function=style(landuse_color(row), 0.75),
        )
        gj.add_child(folium.Tooltip(landuse_tooltip(row)))
        landuse_layer.add_child(gj)
    fmap.add_child(landuse_layer)

    # ---- Слой 2: GSI (коэффициент застройки по пятну) ------------------------ #
    def bib_bbox(value, vmin, vmax, low, high):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "#b6bcbf"
        ratio = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        ratio = min(max(ratio, 0.0), 1.0)
        # Простая интерполяция между low и high цветами
        (a, b, c) = tuple(int(low[i:i + 2], 16) for i in (1, 3, 5))
        (x, y, z) = tuple(int(high[i:i + 2], 16) for i in (1, 3, 5))
        r = int(a + (x - a) * ratio)
        g = int(b + (y - b) * ratio)
        blu = int(c + (z - c) * ratio)
        return f"#{r:02x}{g:02x}{blu:02x}"

    # базовые диапазоны из данных (НЕ отражают новых расчетов)
    gsi_min, gsi_max = float(gdf_wgs["build_ratio"].min()), float(gdf_wgs["build_ratio"].max())
    fsi_min, fsi_max = float(gdf_wgs["fsi"].min()), float(gdf_wgs["fsi"].max())
    pop_min, pop_max = float(gdf_wgs["population_model"].min()), float(gdf_wgs["population_model"].max())
    acc_min, acc_max = float(gdf_wgs["area_accessibility"].min()), float(gdf_wgs["area_accessibility"].max())

    def gsi_tooltip(row):
        return "<br>".join([
            f"<b>Квартал №{int(row['block_id'])}</b>",
            f"GSI (коэф. застройки по пятну): {fmt(row.get('build_ratio'))}",
            f"Площадь пятна застройки: {fmt(row.get('footprint_sum'), 0)} м²",
            f"Площадь квартала: {fmt(row.get('block_area'), 0)} м²",
        ])

    gsi_layer = folium.FeatureGroup(name="GSI (коэф. застройки по пятну)", show=False)
    for _, row in gdf_wgs.iterrows():
        color = bib_bbox(row.get("build_ratio"), gsi_min, gsi_max, "#fdf6e3", "#c0392b")
        gj = folium.GeoJson(
            data={"type": "Feature", "properties": {}, "geometry": row.geometry.__geo_interface__},
            style_function=style(color, 0.75),
        )
        gj.add_child(folium.Tooltip(gsi_tooltip(row)))
        gsi_layer.add_child(gj)
    fmap.add_child(gsi_layer)

    # ---- Слой 3: FSI --------------------------------------------------------- #
    def fsi_tooltip(row):
        return "<br>".join([
            f"<b>Квартал №{int(row['block_id'])}</b>",
            f"FSI / FAR: {fmt(row.get('fsi'))}",
            f"Поэтажная площадь: {fmt(row.get('bfloor_sum'), 0)} м²",
            f"Площадь квартала: {fmt(row.get('block_area'), 0)} м²",
        ])

    fsi_layer = folium.FeatureGroup(name="FSI / FAR ( плотность по этажности)", show=False)
    for _, row in gdf_wgs.iterrows():
        color = bib_bbox(row.get("fsi"), fsi_min, fsi_max, "#eaf4ff", "#1f4e79")
        gj = folium.GeoJson(
            data={"type": "Feature", "properties": {}, "geometry": row.geometry.__geo_interface__},
            style_function=style(color, 0.75),
        )
        gj.add_child(folium.Tooltip(fsi_tooltip(row)))
        fsi_layer.add_child(gj)
    fmap.add_child(fsi_layer)

    # ---- Слой 4: модельное население ----------------------------------------- #
    def pop_tooltip(row):
        return "<br>".join([
            f"<b>Квартал №{int(row['block_id'])}</b>",
            f"Население (модель): {fmt(row.get('population_model'), 0)} чел.",
            "Оценка по этажности зданиями (не официальная статистика)",
        ])

    pop_layer = folium.FeatureGroup(name="Население (модельная оценка)", show=False)
    for _, row in gdf_wgs.iterrows():
        color = bib_bbox(row.get("population_model"), pop_min, pop_max, "#fff7d6", "#8b1a1a")
        gj = folium.GeoJson(
            data={"type": "Feature", "properties": {}, "geometry": row.geometry.__geo_interface__},
            style_function=style(color, 0.75),
        )
        gj.add_child(folium.Tooltip(pop_tooltip(row)))
        pop_layer.add_child(gj)
    fmap.add_child(pop_layer)

    # ---- Слой 5: provision_strong ------------------------------------------- #
    def provision_color(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "#b6bcbf"
        if val < 0.33:
            return "#c0392b"
        if val < 0.67:
            return "#e67e22"
        if val < 1.0:
            return "#f1c40f"
        return "#27ae60"

    def prov_tooltip(row, col):
        return "<br>".join([
            f"<b>Квартал №{int(row['block_id'])}</b>",
            f"provision ({'strong' if col == 'provision_strong' else 'weak'}): {fmt_share(row.get(col))}",
            f"Спрос: {fmt(row.get('demand'), 0)} чел.",
            f"Вместимость школы (восст.): {fmt(row.get('capacity'), 0)} мест",
        ])

    for col, name in [("provision_strong", "Школы: provision_strong"),
                      ("provision_weak", "Школы: provision_weak")]:
        layer = folium.FeatureGroup(name=name, show=False)
        for _, row in gdf_wgs.iterrows():
            color = provision_color(row.get(col))
            gj = folium.GeoJson(
                data={"type": "Feature", "properties": {}, "geometry": row.geometry.__geo_interface__},
                style_function=style(color, 0.75),
            )
            gj.add_child(folium.Tooltip(prov_tooltip(row, col)))
            layer.add_child(gj)
        fmap.add_child(layer)

    # ---- Слой 7: Spacematrix и «Стрелки» ------------------------------------ #
    SPACE_COLORS = {
        "low-rise low-density non-residential": "#f6d55c",
        "mid-rise low-density non-residential": "#947bd3",
    }
    STRELKA_COLORS = {
        "low-rise non-residential": "#6fc2d0",
        "mid-rise non-residential": "#3b6978",
    }

    for color_map, name_col, name in [
        (SPACE_COLORS, "morphotype_spacematrix", "Spacematrix"),
        (STRELKA_COLORS, "morphotype_strelka", "«Стрелки»"),
    ]:
        layer = folium.FeatureGroup(name=f"Морфотип {name}", show=False)
        for _, row in gdf_wgs.iterrows():
            mt = row.get(name_col)
            mt_txt = "нет данных" if mt is None or (isinstance(mt, float) and pd.isna(mt)) else str(mt)
            color = color_map.get(mt_txt, "#b6bcbf")
            gj = folium.GeoJson(
                data={"type": "Feature", "properties": {}, "geometry": row.geometry.__geo_interface__},
                style_function=style(color, 0.75),
            )
            gj.add_child(folium.Tooltip(
                f"<b>Квартал №{int(row['block_id'])}</b><br>{name}: {mt_txt}"
            ))
            layer.add_child(gj)
        fmap.add_child(layer)

    # ---- Слой 8: Area accessibility ---------------------------------------- #
    def acc_tooltip(row):
        return "<br>".join([
            f"<b>Квартал №{int(row['block_id'])}</b>",
            f"Area accessibility: {fmt(row.get('area_accessibility'), 1)} мин",
            "Взвешенное среднее пешеходного времени до других кварталов (5 км/ч)",
        ])

    acc_layer = folium.FeatureGroup(name="Пешеходная доступность (мин, меньше = лучше)", show=False)
    for _, row in gdf_wgs.iterrows():
        # инверт: меньше время = зеленее, больше = краснее
        color = bib_bbox(row.get("area_accessibility"), acc_min, acc_max, "#27ae60", "#c0392b")
        gj = folium.GeoJson(
            data={"type": "Feature", "properties": {}, "geometry": row.geometry.__geo_interface__},
            style_function=style(color, 0.75),
        )
        gj.add_child(folium.Tooltip(acc_tooltip(row)))
        acc_layer.add_child(gj)
    fmap.add_child(acc_layer)

    folium.LayerControl(collapsed=False, position="topleft").add_to(fmap)
    return fmap


def map_to_iframe(fmap) -> str:
    """Рендерит карту в отдельный HTML и вкладывает её как iframe srcdoc (без Jupyter-обёртки)."""
    doc = fmap.get_root().render()          # полный standalone HTML документ
    escaped = html.escape(doc, quote=True)
    return (
        '<iframe class="map-frame" srcdoc="' + escaped + '" '
        'style="width:100%; height:640px; border:0; border-radius:8px;" '
        'loading="lazy" title="Карта кварталов Приморска"></iframe>'
    )


# ----------------------------------------------------------------------------- #
# 4. Графики Plotly (единичные карточки)
# ----------------------------------------------------------------------------- #
def build_charts(gdf_wgs: gpd.GeoDataFrame) -> str:
    df = gdf_wgs.drop(columns=["geometry"]).copy()
    parts = []

    LAND_ORDER = ["RESIDENTIAL", "BUSINESS", "AGRICULTURE", "нет данных"]

    # --- 1. Распределение по землепользованию -------------------------------- #
    lu_counts = (
        df["land_use_name"].fillna("нет данных").value_counts().reindex(LAND_ORDER).fillna(0).astype(int)
    )
    fig_lu = px.bar(
        x=lu_counts.index, y=lu_counts.values,
        labels={"x": "Категория землепользования", "y": "Число кварталов"},
        color=lu_counts.index,
        color_discrete_map={**LANDUSE_COLORS, "нет данных": "#b6bcbf"},
    )
    parts.append(card(fig_lu, "Распределение кварталов по доминирующему землепользованию",
                      "Источник: OSM landuse, разрыв между зонами и кварталами учтён через share_covered / share_no_data."))

    # --- 2. GSI vs FSI -------------------------------------------------------- #
    fig_scatter = px.scatter(
        df, x="build_ratio", y="fsi",
        color="morphotype_spacematrix",
        symbol="morphotype_spacematrix",
        hover_name=df["block_id"].astype(str),
        labels={"build_ratio": "GSI (коэф. застройки по пятну)",
                "fsi": "FSI / FAR (поэтажная площадь / территория)",
                "morphotype_spacematrix": "Морфотип Spacematrix",
                "hover_name": "Квартал №"},
        color_discrete_map={
            "low-rise low-density non-residential": "#f6d55c",
            "mid-rise low-density non-residential": "#947bd3",
        },
    )
    fig_scatter.update_traces(marker=dict(size=11, opacity=0.85, line=dict(width=1, color="#3b4252")))
    fig_scatter.update_layout(legend_title_text="Морфотип Spacematrix")
    parts.append(card(fig_scatter, "Соотношение GSI и FSI по кварталам",
                      "Две оси описывают плотность застройки: GSI — доля площади квартала под пятнами зданий, FSI — отношение общей поэтажной площади к площади квартала."))

    # --- 3. Модель поpopulation ---------------------------------------------- #
    df_pop = df.sort_values("population_model", ascending=False)
    fig_pop = px.bar(
        df_pop, x="block_id", y="population_model",
        labels={"block_id": "№ квартала", "population_model": "Население, чел. (модель)"},
        color_discrete_sequence=["#c0392b"],
    )
    parts.append(card(fig_pop, "Модельная численность населения по кварталам",
                      f"Суммарная модельная оценка: <b>{fmt_int(df['population_model'].sum())} чел.</b> Скорректирована по этажности и площади застройки зданий; <b>не является официальной статистикой</b>."))

    # --- 4. Обеспеченность школами ------------------------------------------- #
    df_prov = df[["block_id", "provision_strong", "provision_weak"]].melt(
        id_vars="block_id", var_name="Методика", value_name="Обеспеченность"
    )
    df_prov["Обеспеченность"] = df_prov["Обеспеченность"] * 100
    df_prov["Методика"] = df_prov["Методика"].map({"provision_strong": "provision_strong", "provision_weak": "provision_weak"})
    fig_prov = px.bar(
        df_prov, x="block_id", y="Обеспеченность", color="Методика", barmode="group",
        labels={"block_id": "№ квартала", "Обеспеченность": "Обеспеченность, %", "Методика": "Методика"},
        color_discrete_map={"provision_strong": "#2e86c1", "provision_weak": "#f39c12"},
    )
    fig_prov.add_hline(y=100, line_dash="dash", line_color="#c0392b",
                       annotation_text="100 % — полная обеспеченность")
    parts.append(card(fig_prov, "Обеспеченность школами по кварталам (модельный расчёт)",
                      "Модельная вместимость школы — 250 мест (восстановлено алгоритмами BlocksNet), в OSM данные о вместимости отсутствуют."))

    # --- 5. Морфотипы Spacematrix -------------------------------------------- #
    sm_counts = df["morphotype_spacematrix"].fillna("нет данных").value_counts()
    sm_counts = sm_counts.sort_values(ascending=False)
    fig_sm = px.bar(
        x=sm_counts.index, y=sm_counts.values,
        labels={"x": "Морфотип", "y": "Число кварталов"},
        color=sm_counts.index,
        color_discrete_map={"low-rise low-density non-residential": "#f6d55c",
                            "mid-rise low-density non-residential": "#947bd3",
                            "нет данных": "#b6bcbf"},
    )
    fig_sm.update_layout(xaxis=dict(tickangle=-20))
    parts.append(card(fig_sm, "Морфотипы Spacematrix (модельная классификация)",
                      "Классификация строится на GSI, FSI и кластеризации; большая часть этажности зданий восстановлена моделью, поэтому результат оценочный."))

    # --- 6. Морфотипы «Стрелки» ---------------------------------------------- #
    st_counts = df["morphotype_strelka"].fillna("нет данных").value_counts()
    st_counts = st_counts.sort_values(ascending=False)
    fig_st = px.bar(
        x=st_counts.index, y=st_counts.values,
        labels={"x": "Тип застройки", "y": "Число кварталов"},
        color=st_counts.index,
        color_discrete_map={"low-rise non-residential": "#6fc2d0",
                            "mid-rise non-residential": "#3b6978",
                            "нет данных": "#b6bcbf"},
    )
    fig_st.update_layout(xaxis=dict(tickangle=-20))
    parts.append(card(fig_st, "Классификация «Стрелки» (модельная классификация)",
                      "Индикаторы GSI и FSI сопоставляются с порогами «Стрелки»; кварталы, где признаки неполны, помечены как «нет данных» (5 из 40)."))

    # --- 7. Пешеходная доступность ------------------------------------------ #
    df_acc = df.sort_values("area_accessibility", ascending=True)
    fig_acc = px.bar(
        df_acc, x="block_id", y="area_accessibility",
        color="area_accessibility", color_continuous_scale=["#27ae60", "#f1c40f", "#c0392b"],
        labels={"block_id": "№ квартала", "area_accessibility": "Время, мин"},
    )
    parts.append(card(fig_acc, "Area accessibility: среднее пешеходное время до остальных кварталов",
                      "Взвешенное среднее по матрице пешеходных времён (скорость 5 км/ч). Меньшее значение — лучше связность."))

    return "\n".join(parts)


def card(fig, title: str, note: str = "") -> str:
    body = pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False})
    note_html = f'<p class="chart-note">{note}</p>' if note else ""
    return f"""<div class="chart-card"><h3>{title}</h3>{body}{note_html}</div>"""


# ----------------------------------------------------------------------------- #
# 5. Таблица с поиском и сортировкой (клиентский JS)
# ----------------------------------------------------------------------------- #
def build_table(df: pd.DataFrame) -> str:
    df = df.copy()
    for col in df.columns:
        if df[col].isna().dtype == bool and df[col].isna().any():
            df[col] = df[col].fillna("нет данных")
    df["sort_id"] = df["№ квартала"].astype(int)

    rows = []
    for _, r in df.sort_values("sort_id").iterrows():
        cells = "".join(f"<td>{v}</td>" for v in r.drop("sort_id"))
        rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(rows)
    headers = "".join(f"<th>{c}</th>" for c in df.columns if c != "sort_id")

    return f"""<div class="table-tools">
        <input id="tbl-search" type="text" placeholder="Поиск по кварталам..." />
    </div>
    <div class="table-wrap">
        <table id="blocks-table" class="blocks-table">
            <thead><tr>{headers}</tr></thead>
            <tbody>{body}</tbody>
        </table>
    </div>
    <script>
    // Поиск
    document.getElementById('tbl-search').addEventListener('keyup', function() {{
        const q = this.value.toLowerCase();
        document.querySelectorAll('#blocks-table tbody tr').forEach(function(tr) {{
            tr.style.display = tr.innerText.toLowerCase().includes(q) ? '' : 'none';
        }});
    }});
    // Сортировка по клику на заголовок
    document.querySelectorAll('#blocks-table thead th').forEach(function(th, idx) {{
        th.addEventListener('click', function() {{
            const tbody = document.querySelector('#blocks-table tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const asc = th.classList.toggle('asc');
            rows.sort(function(a, b) {{
                let av = a.cells[idx].innerText, bv = b.cells[idx].innerText;
                let an = parseFloat(av.replace(/\\s/g, '').replace(',', '.')), bn = parseFloat(bv.replace(/\\s/g, '').replace(',', '.'));
                if (!isNaN(an) && !isNaN(bn)) {{ return asc ? an - bn : bn - an; }}
                return asc ? av.localeCompare(bv) : bv.localeCompare(av);
            }});
            rows.forEach(function(r) {{ tbody.appendChild(r); }});
        }});
    }});
    </script>"""


# ----------------------------------------------------------------------------- #
# 6. Сборка HTML
# ----------------------------------------------------------------------------- #
def build_html(gdf_wgs, charts_html, table_html, map_iframe_html) -> str:
    total_pop = fmt_int(gdf_wgs["population_model"].sum())
    total_demand = fmt_int(gdf_wgs["demand"].sum())
    avg_acc = fmt(gdf_wgs["area_accessibility"].mean(), 1)
    max_fsi = fmt(gdf_wgs["fsi"].max(), 3)
    max_gsi = fmt(gdf_wgs["build_ratio"].max(), 3)
    n_res = int((gdf_wgs["land_use_name"] == "RESIDENTIAL").sum())

    legenda = ""
    for label, color in LEGEND_COLORS.items():
        legenda += f'<div class="legend-item"><span class="legend-color" style="background:{color}"></span>{label}</div>'

    legenda_gsi = '<div class="legend-note">Сплошной цветовой переход: светлое — низкий GSI, тёмно-красное — высокий GSI</div>'
    legenda_prov = '<div class="legend-note">Доли обеспечения: тёмно-красное — &lt;33 %, оранжевое — 33–67 %, жёлтое — 67–100 %, зелёное — 100 % и выше</div>'
    legenda_acc = '<div class="legend-note">Зелёное — хорошая связность (меньшее время), красное — слабая связность</div>'

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Дашборд анализа городской застройки — Приморск (BlocksNet)</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="{PLOTLY_CDN}"></script>
    <style>
      :root {{
         --primary: #2453a6;
         --accent: #f39c12;
         --bg: #f8f9fa;
      }}
      body {{ font-family: 'Segoe UI', Roboto, sans-serif; background: var(--bg); }}
      .main-header {{ background: linear-gradient(90deg, #2453a6 0%, #5da9e9 100%); color:#fff; padding: 2rem 0; }}
      .main-header h1 {{ font-size: 2.1rem; font-weight: 700; }}
      .intro p {{ margin-bottom: 0.5rem; }}
      .card {{ margin: 1.2rem 0; box-shadow: 0 2px 12px rgba(0,0,0,.08); border: 0; }}
      .card-header {{ background: #fff; font-weight: 600; border-bottom: 2px solid var(--primary); }}
      .alert-limit {{ background: #fff8e1; border-left: 5px solid var(--accent); }}
      .chart-card {{ background:#fff; padding:1.2rem; margin:1rem 0; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.06); }}
      .chart-note {{ color:#6b7280; font-size:.9rem; }}
      .legend-item {{ display:inline-flex; align-items:center; margin-right:1.2rem; }}
      .legend-color {{ width:16px; height:12px; display:inline-block; margin-right:.35rem; }}
      .legend-note {{ font-size:.85rem; color:#6b7280; margin-bottom:.5rem; }}
      .map-frame {{ box-shadow: 0 2px 10px rgba(0,0,0,.2); }}
      .blocks-table {{ border-collapse: collapse; width:100%; }}
      .blocks-table th {{ background:#f4f5f7; padding:.45rem .6rem; cursor:pointer; border-bottom:2px solid #dcdfe4; }}
      .blocks-table th:hover {{ background:#e8edf2; }}
      .blocks-table th.asc {{ background:#d8e4f5; }}
      .blocks-table td {{ padding:.4rem .6rem; font-size:.9rem; border-bottom:1px solid #edf0f3; }}
      .blocks-table tr:nth-child(even) {{ background:#fbfcfd; }}
      .blocks-table tr:hover {{ background:#eaf1fa; }}
      .table-tools input {{ padding:.45rem .8rem; border:1px solid #d0d5dd; border-radius:6px; min-width:280px; }}
      .table-wrap {{ max-height:520px; overflow:auto; border:1px solid #e6e9ec; border-radius:8px; }}
    </style>
</head>
<body>
    <header class="main-header intro">
        <div class="container">
            <h1>Городская застройка г. Приморска</h1>
            <p class="lead mb-0">Выборгский район, Ленинградская область</p>
            <p><small>Дашборд построен по результатам расчётов BlocksNet 1.0.0a9, источникам OSM и пешеходного графа (5 км/ч)</small></p>
        </div>
    </header>

    <main class="container">
        <section class="card intro">
            <div class="card-header"><h5>Краткие выводы</h5></div>
            <div class="card-body">
                <ul>
                    <li>Проанализировано <b>40 городских кварталов</b> г. Приморска. Доминирует жилое землепользование ({n_res} квартал(ов) класса RESIDENTIAL).</li>
                    <li>Максимальный коэффициент застройки по пятну (GSI): <b>{max_gsi}</b>; максимальная плотность по FSI/FAR: <b>{max_fsi}</b> — территория преимущественно низкоплотная.</li>
                    <li>Суммарная модельная численность населения: <b>{total_pop} чел.</b></li>
                    <li>Единственная школа имеет восстановленную вместимость 250 мест; суммарный модельный спрос — {total_demand} чел., оценка обеспеченности — по методикам provision_strong / provision_weak.</li>
                    <li>Пешеходная доступность: среднее взвешенное время — <b>{avg_acc} мин</b>.</li>
                </ul>
            </div>
        </section>

        <section class="card alert-limit">
            <div class="card-header"><h5>⚠️ Важные методические ограничения</h5></div>
            <div class="card-body"><ul class="mb-0">
                <li><b>Население 9417 чел.</b> — модельная оценка, а не официальная статистика.</li>
                <li><b>Вместимость школы 250 мест</b> — восстановлена алгоритмом BlocksNet (в OSM атрибут capacity отсутствует).</li>
                <li><b>Исходная этажность зданиями</b> известна только у 135 из 1464 зданий; остальная этажность восстановлена моделью `impute_buildings`.</li>
                <li>Покрытие зонами землепользования OSM неполное — для части кварталов поле no data учтено через <code>share_no_data</code>.</li>
                <li>Пешеходная матрица рассчитана по графу со скоростью 5 км/ч; поле <code>area_accessibility</code> — взвешенное среднее время в минутах (меньше = лучше).</li>
                <li>Классификации Spacematrix / «Стрелки» выполнены по модельным GSI и FSI, поэтому являются оценочными.</li>
            </ul></div>
        </section>

        <section class="card">
            <div class="card-header"><h5>Интерактивная карта кварталов Приморска</h5></div>
            <div class="card-body">
                <p class="mb-2">Находящаяся внутри карта содержит переключаемые тематические слои (меню в верхнем левом углу): землепользование, GSI, FSI, модельное население, provision_strong / provision_weak, Spacematrix, «Стрелки», пешеходная доступность.</p>
                {map_iframe_html}
                <div class="mt-3">
                    <h6>Условные обозначения (землепользование)</h6>
                    {legenda}
                    <h6 class="mt-3">Условные обозначения (GSI и FSI)</h6>
                    {legenda_gsi}
                    <h6>Условные обозначения (обеспеченность школами)</h6>
                    {legenda_prov}
                    <h6>Условные обозначения (пешеходная доступность)</h6>
                    {legenda_acc}
                </div>
            </div>
        </section>

        <section>
            <h2 class="mt-4 mb-2">Графики и распределения</h2>
            {charts_html}
        </section>

        <section class="card mt-4">
            <div class="card-header"><h5>Таблица по всем 40 кварталам</h5></div>
            <div class="card-body">
                {table_html}
                <p class="mt-2 text-muted mb-0">Подсказка: нажмите на заголовок столбца для сортировки; используйте поле поиска для фильтрации кварталов.</p>
            </div>
        </section>

        <section class="card mt-4">
            <div class="card-header"><h5>Источники данных и методика</h5></div>
            <div class="card-body">
                <ul class="mb-0">
                    <li><b>Исходные данные:</b> <a href="https://www.openstreetmap.org/" target="_blank">OpenStreetMap</a> — границы, дороги, ж/д, гидрография, здания, школы, landuse.</li>
                    <li><b>Инструментарий:</b> <a href="https://github.com/aimclub/blocksnet" target="_blank">BlocksNet 1.0.0a9</a> (aimclub/blocksnet); блок pipeline: <a href="https://github.com/aimclub/blocksnet/blob/main/examples/pipeline.ipynb" target="_blank">examples/pipeline.ipynb</a>.</li>
                    <li><b>Пешеходная матрица:</b> graph.graphml (5 км/ч), матрица 40×40 в минутах.</li>
                    <li><b>CRS:</b> EPSG:32635 для расчётов, EPSG:4326 для веб-карты.</li>
                    <li><b>Файл-основание дашборда:</b> <code>data/primorsk_blocksnet/blocks_all_metrics.geojson</code> (40 кварталов, 53 поля), а также <code>blocks_spacematrix.geojson</code> и <code>blocks_strelka.geojson</code>.</li>
                </ul>
            </div>
        </section>
    </main>

    <footer style="background:#2453a6; color:#fff; padding:2rem 0;" class="mt-4">
        <div class="container text-center">
            <p class="mb-0">Дашборд сгенерирован скриптом <code>build_dashboard.py</code></p>
            <p class="mb-0"><small>Практическая работа №2 «ИИ-технологии в урбанистике» | BlocksNet для г. Приморск</small></p>
        </div>
    </footer>
</body>
</html>"""


# ----------------------------------------------------------------------------- #
# 7. Запуск
# ----------------------------------------------------------------------------- #
def main():
    gdf, gdf_wgs, sm, st = load_data()
    print(f"Загружено кварталов: {len(gdf)}")
    print("Строю карту...")
    fmap = build_map(gdf_wgs)
    map_iframe = map_to_iframe(fmap)

    print("Строю графики...")
    charts_html = build_charts(gdf_wgs)

    print("Строю таблицу...")
    df_table = prepare_table(gdf_wgs)
    table_html = build_table(df_table)

    print("Собираю HTML...")
    html_content = build_html(gdf_wgs, charts_html, table_html, map_iframe)

    with io.open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html_content)
    print(f"Сохранено: {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
