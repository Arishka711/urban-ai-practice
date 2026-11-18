"""Интеграционные проверки Streamlit-приложения размещения школ.

Запуск:
    .\.venv\Scripts\python.exe school-placement\test_app.py

Проверяется фактический запуск ``school-placement/app.py`` через
``streamlit.testing.v1.AppTest``: исходный расчёт, ручной сценарий, сброс,
автоматический перебор, сравнение и валидация вместимости. Результаты
сохраняются в ``school-placement/results/app_test_results.csv``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from streamlit.testing.v1 import AppTest  # noqa: E402

import school_model as sm  # noqa: E402

APP_PATH = MODULE_DIR / "app.py"


class AppTests(unittest.TestCase):
    """Проверки пользовательских сценариев приложения."""

    def setUp(self) -> None:
        self.at = AppTest.from_file(str(APP_PATH), default_timeout=180)
        self.at.run()
        self.assert_no_exceptions()

    def assert_no_exceptions(self) -> None:
        self.assertEqual(
            [str(e.value) for e in self.at.exception],
            [],
            msg="Приложение завершилось с ошибкой.",
        )

    def button(self, label: str):
        matches = [b for b in self.at.button if b.label == label]
        self.assertTrue(matches, msg=f"Кнопка не найдена: {label}")
        return matches[0]

    def selectbox(self, label: str):
        matches = [s for s in self.at.selectbox if s.label == label]
        self.assertTrue(matches, msg=f"Выпадающий список не найден: {label}")
        return matches[0]

    # 1. Исходный расчёт --------------------------------------------------

    def test_01_baseline_matches_practice_2(self) -> None:
        metrics = {m.label: m.value for m in self.at.metric}
        self.assertEqual(metrics["Модельное население"], "9 417 чел.")
        self.assertEqual(metrics["Исходная потребность"], "1 148 мест")
        self.assertEqual(metrics["Исходная вместимость"], "250 мест")
        self.assertEqual(metrics["Исходный дефицит"], "898 мест")
        self.assertEqual(metrics["Исходная обеспеченность"], "21,78 %")

        table = self.at.dataframe[0].value
        self.assertEqual(len(table), 40)
        self.assertEqual(int(table["Квартал"].min()), 0)
        self.assertEqual(int(table["Квартал"].max()), 39)
        row_16 = table.loc[table["Квартал"].eq(16)].iloc[0]
        self.assertEqual(int(row_16["Дефицит, мест"]), 238)
        self.assertEqual(row_16["Обеспеченность"], "51,13 %")

    # 2. Ручной сценарий, сумма двух школ и сброс --------------------------

    def test_02_manual_two_schools_and_reset(self) -> None:
        self.button("➕ Добавить школу").click().run(timeout=180)
        self.button("Добавить ещё одну").click().run(timeout=180)
        self.assert_no_exceptions()

        schools = self.at.session_state["schools"]
        self.assertEqual(
            schools,
            [
                {"block_id": 16, "capacity": 600},
                {"block_id": 16, "capacity": 600},
            ],
        )

        self.button("Рассчитать сценарий").click().run(timeout=180)
        self.assert_no_exceptions()
        scenario = self.at.session_state["scenario_result"]
        self.assertIsNotNone(scenario)
        self.assertEqual(scenario.total_capacity, 1450)
        self.assertEqual(scenario.total_demand_left, 288)
        self.assertEqual(
            self.at.session_state["map_mode"],
            "Обеспеченность после строительства",
        )

        table = self.at.dataframe[0].value
        self.assertEqual(len(table), 40)
        self.assertIn("Δ дефицита, мест", table.columns)
        changes = table["Δ дефицита, мест"].astype(str).tolist()
        self.assertTrue(any(value.startswith("-") for value in changes))

        self.button("Сбросить всё").click().run(timeout=180)
        self.assert_no_exceptions()
        self.assertEqual(self.at.session_state["schools"], [])
        self.assertIsNone(self.at.session_state["scenario_result"])
        self.assertIsNone(self.at.session_state["auto_result"])
        self.assertEqual(self.at.session_state["selected_block"], 16)
        self.assertEqual(
            self.at.session_state["map_mode"],
            "Исходная обеспеченность",
        )

        baseline_table = self.at.dataframe[0].value
        self.assertEqual(len(baseline_table), 40)
        self.assertNotIn("Δ дефицита, мест", baseline_table.columns)

    # 3. Автоматический перебор -------------------------------------------

    def test_03_auto_search_checks_all_40_blocks(self) -> None:
        self.button(
            "🔎 Найти оптимальное размещение школы на 600 мест"
        ).click().run(timeout=180)
        self.assert_no_exceptions()

        auto = self.at.session_state["auto_result"]
        self.assertIsNotNone(auto)
        self.assertEqual(auto.attempted_count, 40)
        self.assertEqual(auto.checked_count, 40)
        self.assertTrue(auto.complete)
        self.assertEqual(auto.best_total_deficit, 298)
        self.assertEqual(auto.best_reduction, 600)
        self.assertEqual(auto.best_block_ids, [1, 2, 16, 27, 28, 36, 37])
        self.assertTrue(set(range(40)).issubset(set(auto.ranking["block_id"])))

        metrics = {m.label: m.value for m in self.at.metric}
        self.assertEqual(metrics["Проверено кварталов"], "40 из 40")
        self.assertEqual(metrics["Лучший остаточный дефицит"], "298 мест")
        self.assertEqual(metrics["Равноценных вариантов"], "7")

        ranking = self.at.dataframe[1].value
        self.assertEqual(len(ranking), 40)
        self.assertEqual(int(ranking["Ранг"].min()), 1)
        self.assertTrue(
            ranking.loc[
                ranking["Оптимальный"].eq("да"), "Квартал"
            ].tolist()
            == [1, 2, 16, 27, 28, 36, 37]
        )

    # 4. Сравнение не подменяет выбранные кварталы -------------------------

    def test_04_comparison_uses_selected_blocks(self) -> None:
        self.selectbox("Сценарий А: квартал").select("Квартал 4").run(
            timeout=180
        )
        self.selectbox("Сценарий Б: квартал 1").select("Квартал 16").run(
            timeout=180
        )
        self.selectbox("Сценарий Б: квартал 2").select("Квартал 16").run(
            timeout=180
        )

        self.button(
            "Выполнить автоматический подбор для сценария В"
        ).click().run(timeout=180)
        self.assert_no_exceptions()

        comparison = self.at.dataframe[-1].value
        self.assertEqual(len(comparison), 4)
        self.assertTrue(
            comparison.loc[
                comparison["Сценарий"].str.startswith("А:"), "Кварталы"
            ].iloc[0]
            == "Квартал 4"
        )
        self.assertTrue(
            comparison.loc[
                comparison["Сценарий"].str.startswith("Б:"), "Кварталы"
            ].iloc[0]
            == "Квартал 16"
        )
        c_row = comparison.loc[comparison["Сценарий"].str.startswith("В:")].iloc[0]
        self.assertIn("Квартал 16", str(c_row["Кварталы"]))
        self.assertIn("один из лучших", str(c_row["Кварталы"]))
        self.assertEqual(self.selectbox("Сценарий А: квартал").value, 4)

    # 5. Вместимость -------------------------------------------------------

    def test_05_invalid_capacity_is_rejected(self) -> None:
        number_input = self.at.number_input[0]
        self.assertEqual(number_input.min, 1.0)
        number_input.set_value(0).run(timeout=180)
        self.assert_no_exceptions()
        self.assertGreaterEqual(self.at.number_input[0].value, 1)

        for bad in (0, -1, 1.5, "300", None, True):
            with self.assertRaises(sm.InvalidCapacityError):
                sm.validate_capacity(bad)


# ---------------------------------------------------------------------------
# Сохранение результатов
# ---------------------------------------------------------------------------


class CsvTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.rows: list[dict[str, str]] = []

    def _record(self, test, status: str, detail: str = "") -> None:
        self.rows.append(
            {
                "test": test.id(),
                "status": status,
                "detail": " ".join(str(detail).split())[:4000],
            }
        )

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "PASS")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "FAIL", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "ERROR", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "SKIP", reason)


def main() -> int:
    from loguru import logger

    logger.remove()  # не засорять вывод служебными сообщениями BlocksNet

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AppTests)
    runner = unittest.TextTestRunner(
        verbosity=2, resultclass=CsvTestResult
    )
    result = runner.run(suite)

    out_dir = sm.DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "app_test_results.csv"
    pd.DataFrame(result.rows).to_csv(csv_path, index=False)
    print()
    print(f"Результаты проверок приложения сохранены: {csv_path}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
