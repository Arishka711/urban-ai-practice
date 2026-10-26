import io
import os

p = "dashboard.html"
assert os.path.exists(p), "dashboard.html не существует"
content = io.open(p, encoding="utf-8").read()
size_kb = os.path.getsize(p) / 1024
print(f"SIZE {size_kb:.0f} KB")

required = [
    "Интерактивная карта кварталов Приморска",
    "srcdoc",
    "Краткие выводы",
    "Важные методические ограничения",
    "9417",
    "250 мест",
    "135 из 1464",
    "provision_strong",
    "provision_weak",
    "Spacematrix",
    "Стрелки",
    "GSI",
    "FSI",
    "area_accessibility",
    "Землепользование",
    "Таблица по всем 40 кварталам",
    "tbl-search",
    "Источники данных и методика",
    "plotly",
    "blocks_all_metrics.geojson",
]
missing = [r for r in required if r not in content]
print("MISSING:", missing if missing else "none")

# Проверка, что нет артефактов Jupyter ("Make this Notebook Trusted")
print("JUPYTER_TRUST_MSG:", "Make this Notebook Trusted" in content)

# Счёт слоёв карты (FeatureGroup)
print("FEATURE_GROUP_COUNT:", content.count("FeatureGroup"))

# Счёт участков таблицы
print("TABLE_ROWS:", content.count("<tbody>") and content.split("<tbody>")[1].split("</tbody>")[0].count("<tr>"))

print("CHECKS_DONE")
