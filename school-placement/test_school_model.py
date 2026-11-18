"""Тесты расчётного модуля размещения школ.

Запуск:
    .\.venv\Scripts\python.exe school-placement\test_school_model.py

Тесты используют только сохранённые данные практики № 2 и фактический
BlocksNet 1.0.0a9. Результаты тестов сохраняются в
``school-placement/results/test_results.csv``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import school_model as sm  # noqa: E402


class SchoolModelTests(unittest.TestCase):
    """Проверки baseline, сценариев, автоматического поиска и ошибок."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = sm.load_model()
        cls.baseline = cls.model.baseline
        cls.auto = cls.model.auto_search_one_school(600)

    # 1. Исходное состояние ------------------------------------------------

    def test_01_baseline_matches_saved_result(self) -> None:
        self.assertEqual(self.baseline.total_population, 9417)
        self.assertEqual(self.baseline.total_demand, 1148)
        self.assertEqual(self.baseline.total_capacity, 250)
        self.assertEqual(self.baseline.total_demand_within, 250)
        self.assertEqual(self.baseline.total_demand_left, 898)
        self.assertAlmostEqual(
            self.baseline.aggregate_provision, 250 / 1148, places=12
        )
        self.assertAlmostEqual(
            float(self.baseline.provision_strong[16]), 249 / 487, places=9
        )
        self.assertEqual(float(self.baseline.provision_strong[21]), 1.0)
        self.assertTrue(np.isnan(self.baseline.provision_strong[11]))

    # 2. Одна школа --------------------------------------------------------

    def test_02_one_school_600_increases_capacity_to_850(self) -> None:
        scenario = self.model.place_one_school(0, 600)
        self.assertEqual(scenario.total_capacity, 850)
        self.assertEqual(int(scenario.capacity[0]), 600)
        self.assertEqual(scenario.total_demand, 1148)
        self.assertEqual(scenario.total_population, 9417)
        self.assertEqual(int(scenario.capacity.sum()), 850)

    # 3. Две школы ---------------------------------------------------------

    def test_03_two_schools_300_each_give_850(self) -> None:
        same_block = self.model.place_two_schools(0, 300, 0, 300)
        self.assertEqual(same_block.total_capacity, 850)
        self.assertEqual(int(same_block.capacity[0]), 600)

        two_blocks = self.model.place_two_schools(0, 300, 1, 300)
        self.assertEqual(two_blocks.total_capacity, 850)
        self.assertEqual(int(two_blocks.capacity[0]), 300)
        self.assertEqual(int(two_blocks.capacity[1]), 300)

        one_600 = self.model.place_one_school(0, 600)
        np.testing.assert_allclose(
            same_block.provision_strong,
            one_600.provision_strong,
            equal_nan=True,
        )
        self.assertEqual(
            same_block.total_demand_left, one_600.total_demand_left
        )

    # 4. Исходные данные не изменяются ------------------------------------

    def test_04_repeated_calculations_do_not_change_source_data(self) -> None:
        before_capacity = self.model.base_capacity
        before_population = self.model.population
        before_baseline = self.baseline.to_frame()

        self.model.place_one_school(5, 600)
        self.model.place_two_schools(6, 300, 7, 300)
        self.model.auto_search_one_school(300)

        np.testing.assert_array_equal(self.model.base_capacity, before_capacity)
        np.testing.assert_array_equal(self.model.population, before_population)
        pd.testing.assert_frame_equal(self.baseline.to_frame(), before_baseline)

        # Свойства возвращают копии: изменение копии не меняет модель.
        copy = self.model.base_capacity
        copy[0] = 9999
        self.assertEqual(int(self.model.base_capacity[0]), 0)

    # 5. Валидация вместимости --------------------------------------------

    def test_05_invalid_capacity_and_block_are_rejected(self) -> None:
        for bad in (0, -1, 1.5, 300.0, "300", None, True):
            with self.assertRaises(
                sm.InvalidCapacityError, msg=f"должно быть отклонено: {bad!r}"
            ):
                self.model.place_one_school(0, bad)

        for bad_block in (40, -1, 1.0, "1", True):
            with self.assertRaises(
                sm.InvalidBlockError,
                msg=f"должен быть отклонён квартал: {bad_block!r}",
            ):
                self.model.place_one_school(bad_block, 300)

        valid = self.model.place_one_school(np.int64(0), np.int64(300))
        self.assertEqual(int(valid.capacity[0]), 300)

    # 6. Автоматический перебор -------------------------------------------

    def test_06_auto_search_checks_all_40_blocks(self) -> None:
        self.assertEqual(self.auto.attempted_count, 40)
        self.assertEqual(self.auto.checked_count, 40)
        self.assertTrue(self.auto.complete)
        self.assertEqual(self.auto.capacity, 600)

        checked = set(int(x) for x in self.auto.ranking["block_id"])
        self.assertEqual(checked, set(range(40)))
        for zero_population_block in (11, 17, 18, 19, 30, 37):
            self.assertIn(zero_population_block, checked)

        self.assertTrue((self.auto.ranking["total_capacity"] == 850).all())
        self.assertTrue(
            self.auto.ranking["deficit_reduction"].is_monotonic_decreasing
        )
        self.assertEqual(self.auto.best_total_deficit, 298)
        self.assertEqual(self.auto.best_reduction, 600)
        self.assertGreaterEqual(len(self.auto.best_block_ids), 1)

    # 7. Ручной расчёт совпадает с автоматическим --------------------------

    def test_07_manual_calculation_matches_auto_for_best_blocks(self) -> None:
        for block_id in self.auto.best_block_ids:
            manual = self.model.place_one_school(block_id, 600)
            auto_result = self.auto.scenario_results[block_id]

            self.assertEqual(manual.total_demand_left, self.auto.best_total_deficit)
            self.assertEqual(
                manual.total_demand_left, auto_result.total_demand_left
            )
            np.testing.assert_allclose(
                manual.provision_strong,
                auto_result.provision_strong,
                equal_nan=True,
            )

            row = self.auto.ranking.loc[
                self.auto.ranking["block_id"] == block_id
            ].iloc[0]
            self.assertEqual(
                int(row["total_demand_left"]), manual.total_demand_left
            )
            self.assertEqual(
                int(row["deficit_reduction"]), self.auto.best_reduction
            )

    # 8. Ошибки и недостижимые пары не замалчиваются -----------------------

    def test_08_errors_are_reported_and_not_replaced_with_zero(self) -> None:
        matrix = self.model.accessibility_matrix
        self.assertEqual(int(matrix.isna().sum().sum()), 0)

        with self.assertRaises(sm.InvalidBlockError):
            self.model.place_one_school(40, 600)

        original_compute = self.model._compute_result

        def failing_compute(capacity, context=None):
            if context == 11:
                raise RuntimeError("simulated BlocksNet failure")
            return original_compute(capacity, context=context)

        self.model._compute_result = failing_compute
        try:
            partial = self.model.auto_search_one_school(600)
        finally:
            del self.model._compute_result

        self.assertEqual(partial.attempted_count, 40)
        self.assertEqual(partial.checked_count, 39)
        self.assertFalse(partial.complete)
        self.assertTrue(
            any(error["block_id"] == 11 for error in partial.errors)
        )
        self.assertNotIn(11, set(int(x) for x in partial.ranking["block_id"]))
        self.assertTrue((partial.ranking["total_capacity"] == 850).all())
        self.assertNotIn(11, partial.scenario_results)

    # 9. Матрица и идентификаторы -----------------------------------------

    def test_09_matrix_ids_and_values_are_consistent(self) -> None:
        matrix = self.model.accessibility_matrix
        self.assertEqual(matrix.shape, (40, 40))
        np.testing.assert_array_equal(
            matrix.index.to_numpy(), np.arange(40)
        )
        np.testing.assert_array_equal(
            matrix.columns.to_numpy(), np.arange(40)
        )
        np.testing.assert_allclose(np.diag(matrix.to_numpy()), 0.0)
        self.assertTrue(
            np.allclose(matrix.to_numpy(), matrix.to_numpy().T)
        )


# ---------------------------------------------------------------------------
# Сохранение результатов тестов в CSV
# ---------------------------------------------------------------------------


class CsvTestResult(unittest.TextTestResult):
    """TextTestResult, который запоминает результаты для CSV."""

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

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SchoolModelTests)
    runner = unittest.TextTestRunner(
        verbosity=2, resultclass=CsvTestResult
    )
    result = runner.run(suite)

    out_dir = sm.DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "test_results.csv"
    pd.DataFrame(result.rows).to_csv(csv_path, index=False)

    print()
    print(f"Результаты тестов сохранены: {csv_path}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
