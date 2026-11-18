"""Расчётный модуль для оценки размещения новых школ в г. Приморск.

Модуль использует только сохранённые данные практики № 2 и фактический API
BlocksNet 1.0.0a9 (``competitive_provision``). Собственная упрощённая формула
обеспеченности не используется.

Основные сущности:
    load_model()                 — загрузка данных и воспроизведение baseline;
    SchoolModel.baseline         — исходная обеспеченность без новых школ;
    SchoolModel.place_one_school — размещение одной новой школы;
    SchoolModel.place_two_schools— размещение двух новых школ;
    SchoolModel.auto_search_one_school — перебор всех 40 кварталов.

Все расчёты выполняются на копии вектора вместимости. Исходные население,
спрос и базовая вместимость не изменяются.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

from blocksnet.analysis.provision import competitive_provision

# ---------------------------------------------------------------------------
# Пути и параметры модели
# ---------------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "primorsk_blocksnet"
DEFAULT_RESULTS_DIR = MODULE_DIR / "results"

BLOCK_COUNT = 40
BLOCK_IDS = tuple(range(BLOCK_COUNT))

ACCESSIBILITY_MINUTES = 15
DEMAND_PER_1000 = 120

BASELINE_POPULATION = 9417
BASELINE_DEMAND = 1148
BASELINE_CAPACITY = 250
BASELINE_DEFICIT = 898

_REQUIRED_METRIC_COLUMNS = (
    "block_id",
    "population_model",
    "schools_capacity",
    "capacity",
    "demand",
    "demand_left",
    "demand_within",
    "demand_without",
    "capacity_left",
    "capacity_within",
    "capacity_without",
    "provision_strong",
    "provision_weak",
)

_PROVISION_FIELDS = (
    "demand",
    "capacity",
    "demand_left",
    "demand_within",
    "demand_without",
    "capacity_left",
    "capacity_within",
    "capacity_without",
    "provision_strong",
    "provision_weak",
)

_INT_FIELDS = {
    "demand",
    "capacity",
    "demand_left",
    "demand_within",
    "demand_without",
    "capacity_left",
    "capacity_within",
    "capacity_without",
}


# ---------------------------------------------------------------------------
# Исключения
# ---------------------------------------------------------------------------


class SchoolModelError(Exception):
    """Базовая ошибка расчётного модуля."""


class DataValidationError(SchoolModelError):
    """Исходные данные не соответствуют ожидаемой структуре."""


class InvalidBlockError(SchoolModelError, ValueError):
    """Некорректный идентификатор квартала."""


class InvalidCapacityError(SchoolModelError, ValueError):
    """Некорректная вместимость новой школы."""


class ProvisionCalculationError(SchoolModelError, RuntimeError):
    """Ошибка расчёта обеспеченности в BlocksNet."""


# ---------------------------------------------------------------------------
# Валидация и вспомогательные функции
# ---------------------------------------------------------------------------


def _readonly(array: np.ndarray) -> np.ndarray:
    """Возвращает копию массива, доступную только для чтения."""
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def validate_block_id(block_id: Any, valid_ids: Iterable[int] = BLOCK_IDS) -> int:
    """Проверяет идентификатор квартала и возвращает int."""
    if isinstance(block_id, bool) or not isinstance(block_id, (int, np.integer)):
        raise InvalidBlockError(
            f"Идентификатор квартала должен быть целым числом, получено: {block_id!r}"
        )
    value = int(block_id)
    if value not in set(int(x) for x in valid_ids):
        raise InvalidBlockError(
            f"Квартал {value} отсутствует. Допустимые ID: 0…{BLOCK_COUNT - 1}."
        )
    return value


def validate_capacity(value: Any, field_name: str = "capacity") -> int:
    """Разрешает только положительные целые числа мест."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise InvalidCapacityError(
            f"{field_name}: вместимость должна быть целым числом, получено: {value!r}"
        )
    value = int(value)
    if value <= 0:
        raise InvalidCapacityError(
            f"{field_name}: вместимость должна быть положительной, получено: {value}"
        )
    return value


def _validate_capacity_vector(capacity: np.ndarray) -> np.ndarray:
    array = np.asarray(capacity)
    if array.shape != (BLOCK_COUNT,):
        raise InvalidCapacityError(
            f"Вектор вместимости должен иметь длину {BLOCK_COUNT}, получено: {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise InvalidCapacityError("Вектор вместимости должен содержать целые числа.")
    if np.any(array < 0):
        raise InvalidCapacityError("Вместимость не может быть отрицательной.")
    return array.astype(np.int64, copy=False)


# ---------------------------------------------------------------------------
# Результат расчёта
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvisionResult:
    """Результат одного расчёта обеспеченности."""

    block_ids: np.ndarray
    population: np.ndarray
    demand: np.ndarray
    capacity: np.ndarray
    demand_left: np.ndarray
    demand_within: np.ndarray
    demand_without: np.ndarray
    capacity_left: np.ndarray
    capacity_within: np.ndarray
    capacity_without: np.ndarray
    provision_strong: np.ndarray
    provision_weak: np.ndarray
    links: pd.DataFrame

    def __post_init__(self) -> None:
        for name in (
            "block_ids",
            "population",
            "demand",
            "capacity",
            "demand_left",
            "demand_within",
            "demand_without",
            "capacity_left",
            "capacity_within",
            "capacity_without",
            "provision_strong",
            "provision_weak",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "links", self.links.copy())

    @property
    def total_population(self) -> int:
        return int(round(float(self.population.sum())))

    @property
    def total_demand(self) -> int:
        return int(self.demand.sum())

    @property
    def total_capacity(self) -> int:
        return int(self.capacity.sum())

    @property
    def total_demand_left(self) -> int:
        """Суммарный остаточный дефицит мест."""
        return int(self.demand_left.sum())

    @property
    def total_demand_within(self) -> int:
        return int(self.demand_within.sum())

    @property
    def aggregate_provision(self) -> float:
        """Обеспеченность территории: sum(demand_within) / sum(demand)."""
        if self.total_demand == 0:
            return float("nan")
        return self.total_demand_within / self.total_demand

    def to_frame(self) -> pd.DataFrame:
        """Таблица по кварталам (копия, пригодная для CSV)."""
        return pd.DataFrame(
            {
                "block_id": self.block_ids.astype(int),
                "population": self.population,
                "demand": self.demand,
                "capacity": self.capacity,
                "demand_left": self.demand_left,
                "demand_within": self.demand_within,
                "demand_without": self.demand_without,
                "capacity_left": self.capacity_left,
                "capacity_within": self.capacity_within,
                "capacity_without": self.capacity_without,
                "provision_strong": self.provision_strong,
                "provision_weak": self.provision_weak,
            }
        )

    def compare_with(self, other: "ProvisionResult") -> pd.DataFrame:
        """Сравнение с другим результатом по кварталам."""
        if not np.array_equal(self.block_ids, other.block_ids):
            raise ValueError("Нельзя сравнивать результаты с разными наборами кварталов.")
        return pd.DataFrame(
            {
                "block_id": self.block_ids.astype(int),
                "population": self.population,
                "demand": self.demand,
                "capacity_before": self.capacity,
                "capacity_after": other.capacity,
                "capacity_change": other.capacity - self.capacity,
                "demand_left_before": self.demand_left,
                "demand_left_after": other.demand_left,
                "deficit_change": other.demand_left - self.demand_left,
                "provision_before": self.provision_strong,
                "provision_after": other.provision_strong,
                "provision_change": other.provision_strong - self.provision_strong,
            }
        )


@dataclass(frozen=True)
class AutoSearchResult:
    """Результат автоматического перебора кварталов."""

    capacity: int
    baseline: ProvisionResult
    ranking: pd.DataFrame
    best_block_ids: list[int]
    best_total_deficit: int
    best_reduction: int
    attempted_count: int
    checked_count: int
    scenario_results: dict[int, ProvisionResult]
    errors: list[dict[str, Any]]
    complete: bool

    @property
    def best_rows(self) -> pd.DataFrame:
        return self.ranking.loc[self.ranking["is_best"]].copy()

    def summary(self) -> str:
        best = ", ".join(str(x) for x in self.best_block_ids)
        return (
            f"Вместимость новой школы: {self.capacity} мест; "
            f"проверено: {self.checked_count}/{self.attempted_count}; "
            f"лучшие кварталы: {best}; "
            f"остаточный дефицит: {self.best_total_deficit}; "
            f"сокращение: {self.best_reduction}; "
            f"полный перебор: {'да' if self.complete else 'нет'}"
        )


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------


class SchoolModel:
    """Модель обеспеченности школами на сохранённых данных практики № 2."""

    def __init__(
        self,
        *,
        block_ids: np.ndarray,
        population: np.ndarray,
        base_capacity: np.ndarray,
        reference: dict[str, np.ndarray],
        accessibility_matrix: pd.DataFrame,
        data_dir: Path,
    ) -> None:
        self._block_ids = _readonly(np.asarray(block_ids, dtype=int))
        self._population = _readonly(np.asarray(population, dtype=float))
        self._base_capacity = _readonly(np.asarray(base_capacity, dtype=int))
        self._reference = {
            name: _readonly(np.asarray(values))
            for name, values in reference.items()
        }
        self._accessibility_matrix = accessibility_matrix.copy()
        self.data_dir = Path(data_dir)
        self._baseline: ProvisionResult | None = None

    # -- свойства ---------------------------------------------------------

    @property
    def block_ids(self) -> np.ndarray:
        return np.array(self._block_ids, copy=True)

    @property
    def population(self) -> np.ndarray:
        return np.array(self._population, copy=True)

    @property
    def base_capacity(self) -> np.ndarray:
        return np.array(self._base_capacity, copy=True)

    @property
    def accessibility_matrix(self) -> pd.DataFrame:
        return self._accessibility_matrix.copy()

    @property
    def baseline(self) -> ProvisionResult:
        if self._baseline is None:
            raise ProvisionCalculationError("Baseline ещё не рассчитан.")
        return self._baseline

    # -- внутренний расчёт -------------------------------------------------

    def _compute_result(
        self, capacity: np.ndarray, context: Any = None
    ) -> ProvisionResult:
        """Один вызов competitive_provision на заданном векторе вместимости."""
        capacity_vector = _validate_capacity_vector(capacity)

        blocks_df = pd.DataFrame(
            {
                "population": self._population,
                "capacity": capacity_vector,
            },
            index=pd.Index(self._block_ids),
        )

        try:
            prov, links = competitive_provision(
                blocks_df,
                self._accessibility_matrix,
                accessibility=ACCESSIBILITY_MINUTES,
                demand=DEMAND_PER_1000,
                self_supply=True,
                max_depth=1,
            )
        except Exception as exc:  # ошибки BlocksNet не подменяем нулями
            where = "" if context is None else f" (вариант: {context})"
            raise ProvisionCalculationError(
                f"competitive_provision завершился ошибкой{where}: {exc}"
            ) from exc

        if not np.array_equal(prov.index.to_numpy(dtype=int), self._block_ids):
            raise ProvisionCalculationError(
                "BlocksNet вернул результат с другим набором или порядком кварталов."
            )

        values: dict[str, np.ndarray] = {}
        for name in _PROVISION_FIELDS:
            if name not in prov.columns:
                raise ProvisionCalculationError(
                    f"BlocksNet не вернул обязательное поле {name!r}."
                )
            if name in _INT_FIELDS:
                values[name] = prov[name].to_numpy(dtype=np.int64, copy=True)
            else:
                values[name] = prov[name].to_numpy(dtype=float, copy=True)

        return ProvisionResult(
            block_ids=self._block_ids,
            population=self._population,
            links=links,
            **values,
        )

    # -- baseline ----------------------------------------------------------

    def verify_baseline(self) -> ProvisionResult:
        """Воспроизводит исходный расчёт и сверяет его с сохранёнными полями."""
        result = self._compute_result(self._base_capacity, context="baseline")

        for name, expected in self._reference.items():
            actual = getattr(result, name)
            if not np.allclose(actual, expected, equal_nan=True):
                mismatch = np.where(
                    ~np.isclose(actual, expected, equal_nan=True)
                )[0]
                raise DataValidationError(
                    "Baseline не воспроизводит сохранённый результат: "
                    f"поле {name!r}, кварталы {mismatch.tolist()[:10]}, "
                    f"максимальное расхождение "
                    f"{np.nanmax(np.abs(actual - expected))!r}."
                )

        if result.total_population != BASELINE_POPULATION:
            raise DataValidationError(
                f"Суммарное население {result.total_population}, "
                f"ожидалось {BASELINE_POPULATION}."
            )
        if result.total_demand != BASELINE_DEMAND:
            raise DataValidationError(
                f"Суммарный спрос {result.total_demand}, ожидалось {BASELINE_DEMAND}."
            )
        if result.total_capacity != BASELINE_CAPACITY:
            raise DataValidationError(
                f"Суммарная вместимость {result.total_capacity}, "
                f"ожидалось {BASELINE_CAPACITY}."
            )
        if result.total_demand_left != BASELINE_DEFICIT:
            raise DataValidationError(
                f"Остаточный дефицит {result.total_demand_left}, "
                f"ожидалось {BASELINE_DEFICIT}."
            )

        self._baseline = result
        return result

    # -- сценарии ----------------------------------------------------------

    def place_schools(
        self, placements: Sequence[tuple[int, int]]
    ) -> ProvisionResult:
        """Размещает произвольный набор школ, начиная от исходной вместимости."""
        items = list(placements)
        capacity = self._base_capacity.copy()
        if not items:
            return self._compute_result(capacity, context="без новых школ")

        normalized: list[tuple[int, int]] = []
        for block_id, places in items:
            bid = validate_block_id(block_id, self._block_ids)
            value = validate_capacity(places)
            capacity[bid] += value
            normalized.append((bid, value))

        return self._compute_result(capacity, context=normalized)

    def place_one_school(self, block_id: int, capacity: int) -> ProvisionResult:
        """Сценарий Б: одна новая школа в квартале."""
        return self.place_schools([(block_id, capacity)])

    def place_two_schools(
        self,
        block_id_1: int,
        capacity_1: int,
        block_id_2: int,
        capacity_2: int,
    ) -> ProvisionResult:
        """Сценарий В: две новые школы; вместимости суммируются в одном квартале."""
        return self.place_schools(
            [(block_id_1, capacity_1), (block_id_2, capacity_2)]
        )

    # -- автоматический подбор --------------------------------------------

    def auto_search_one_school(
        self,
        capacity: int = 600,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> AutoSearchResult:
        """Перебирает все 40 кварталов для одной школы заданной вместимости.

        ``progress_callback(checked, total, block_id)`` вызывается после каждого
        проверенного квартала; нужен для индикатора выполнения в интерфейсе.
        """
        places = validate_capacity(capacity)

        rows: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        scenario_results: dict[int, ProvisionResult] = {}
        attempted_count = len(self._block_ids)

        for position, block_id in enumerate(self._block_ids, start=1):
            block_id = int(block_id)
            # Каждый вариант начинается строго от исходной вместимости.
            capacity_vector = self._base_capacity.copy()
            capacity_vector[block_id] += places

            try:
                result = self._compute_result(capacity_vector, context=block_id)
            except Exception as exc:
                errors.append(
                    {
                        "block_id": block_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                if progress_callback is not None:
                    progress_callback(position, attempted_count, block_id)
                continue

            scenario_results[block_id] = result
            rows.append(
                {
                    "block_id": block_id,
                    "new_capacity_in_block": int(
                        self._base_capacity[block_id] + places
                    ),
                    "total_capacity": result.total_capacity,
                    "total_demand": result.total_demand,
                    "total_demand_left": result.total_demand_left,
                    "deficit_reduction": (
                        self.baseline.total_demand_left
                        - result.total_demand_left
                    ),
                    "aggregate_provision": result.aggregate_provision,
                    "aggregate_provision_change": (
                        result.aggregate_provision
                        - self.baseline.aggregate_provision
                    ),
                }
            )
            if progress_callback is not None:
                progress_callback(position, attempted_count, block_id)

        if not rows:
            raise ProvisionCalculationError(
                "Автоматический поиск не смог рассчитать ни одного квартала. "
                f"Ошибки: {errors}"
            )

        ranking = pd.DataFrame(rows)
        max_reduction = int(ranking["deficit_reduction"].max())
        ranking["is_best"] = ranking["deficit_reduction"].eq(max_reduction)
        ranking = ranking.sort_values(
            ["deficit_reduction", "total_demand_left", "block_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        ranking["rank"] = (
            ranking["deficit_reduction"]
            .rank(method="min", ascending=False)
            .astype(int)
        )

        best_rows = ranking.loc[ranking["is_best"]]
        best_block_ids = sorted(
            int(x) for x in best_rows["block_id"].tolist()
        )
        best_total_deficit = int(best_rows["total_demand_left"].min())
        best_reduction = int(best_rows["deficit_reduction"].max())

        attempted_count = len(self._block_ids)
        checked_count = len(ranking)
        complete = checked_count == attempted_count and not errors

        return AutoSearchResult(
            capacity=places,
            baseline=self.baseline,
            ranking=ranking,
            best_block_ids=best_block_ids,
            best_total_deficit=best_total_deficit,
            best_reduction=best_reduction,
            attempted_count=attempted_count,
            checked_count=checked_count,
            scenario_results=scenario_results,
            errors=errors,
            complete=complete,
        )


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------


def load_model(
    data_dir: str | Path | None = None, verify_baseline: bool = True
) -> SchoolModel:
    """Загружает сохранённые данные и воспроизводит исходную обеспеченность."""
    path = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    metrics_path = path / "blocks_all_metrics.geojson"
    matrix_path = path / "accessibility_matrix_blocks_minutes.csv"

    for required_path in (metrics_path, matrix_path):
        if not required_path.exists():
            raise DataValidationError(f"Не найден обязательный файл: {required_path}")

    metrics = gpd.read_file(metrics_path)
    missing = [c for c in _REQUIRED_METRIC_COLUMNS if c not in metrics.columns]
    if missing:
        raise DataValidationError(
            f"В {metrics_path.name} отсутствуют поля: {missing}"
        )
    if len(metrics) != BLOCK_COUNT:
        raise DataValidationError(
            f"Ожидалось {BLOCK_COUNT} кварталов, получено {len(metrics)}."
        )

    metrics = metrics.sort_values("block_id").reset_index(drop=True)
    block_ids = metrics["block_id"].to_numpy(dtype=np.int64, copy=True)
    if not np.array_equal(block_ids, np.arange(BLOCK_COUNT)):
        raise DataValidationError(
            "Идентификаторы кварталов должны быть 0…39 без пропусков и дублей."
        )

    population = metrics["population_model"].to_numpy(dtype=float, copy=True)
    if np.any(~np.isfinite(population)) or np.any(population < 0):
        raise DataValidationError("Население содержит NaN или отрицательные значения.")

    capacity_float = metrics["schools_capacity"].to_numpy(dtype=float, copy=True)
    if np.any(~np.isfinite(capacity_float)) or np.any(capacity_float < 0):
        raise DataValidationError("Вместимость содержит NaN или отрицательные значения.")
    if not np.allclose(capacity_float, np.rint(capacity_float)):
        raise DataValidationError("Вместимость должна быть целым числом мест.")
    base_capacity = capacity_float.astype(np.int64)

    if not np.array_equal(
        metrics["capacity"].to_numpy(dtype=np.int64), base_capacity
    ):
        raise DataValidationError(
            "Поля schools_capacity и capacity в blocks_all_metrics различаются."
        )

    reference: dict[str, np.ndarray] = {}
    for name in _PROVISION_FIELDS:
        if name == "capacity":
            reference[name] = base_capacity
        elif name in _INT_FIELDS:
            reference[name] = metrics[name].to_numpy(dtype=np.int64, copy=True)
        else:
            reference[name] = metrics[name].to_numpy(dtype=float, copy=True)

    if int(round(population.sum())) != BASELINE_POPULATION:
        raise DataValidationError(
            f"Суммарное население {population.sum():.0f}, "
            f"ожидалось {BASELINE_POPULATION}."
        )
    if int(base_capacity.sum()) != BASELINE_CAPACITY:
        raise DataValidationError(
            f"Суммарная вместимость {base_capacity.sum()}, "
            f"ожидалось {BASELINE_CAPACITY}."
        )
    if int(reference["demand"].sum()) != BASELINE_DEMAND:
        raise DataValidationError(
            f"Суммарный спрос {reference['demand'].sum()}, "
            f"ожидалось {BASELINE_DEMAND}."
        )
    if int(reference["demand_left"].sum()) != BASELINE_DEFICIT:
        raise DataValidationError(
            f"Остаточный дефицит {reference['demand_left'].sum()}, "
            f"ожидалось {BASELINE_DEFICIT}."
        )

    matrix = pd.read_csv(matrix_path, index_col=0)
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)
    if matrix.shape != (BLOCK_COUNT, BLOCK_COUNT):
        raise DataValidationError(
            f"Матрица доступности должна быть {BLOCK_COUNT}×{BLOCK_COUNT}, "
            f"получено {matrix.shape}."
        )
    if set(matrix.index) != set(BLOCK_IDS) or set(matrix.columns) != set(BLOCK_IDS):
        raise DataValidationError(
            "Строки/столбцы матрицы не совпадают с идентификаторами кварталов."
        )
    matrix = matrix.loc[list(BLOCK_IDS), list(BLOCK_IDS)]
    if matrix.isna().any().any():
        raise DataValidationError(
            "В матрице доступности есть NaN (недостижимые пары). "
            "Для таких пар нужна явная обработка, а не замена нулём."
        )
    if not np.allclose(np.diag(matrix.to_numpy(dtype=float)), 0.0):
        raise DataValidationError("Диагональ матрицы доступности должна быть нулевой.")

    model = SchoolModel(
        block_ids=block_ids,
        population=population,
        base_capacity=base_capacity,
        reference=reference,
        accessibility_matrix=matrix,
        data_dir=path,
    )
    if verify_baseline:
        model.verify_baseline()
    return model


# ---------------------------------------------------------------------------
# Сохранение результатов
# ---------------------------------------------------------------------------


def write_results(
    model: SchoolModel | None = None,
    output_dir: str | Path | None = None,
    capacity: int = 600,
) -> AutoSearchResult:
    """Сохраняет baseline и рейтинг автоматического поиска в CSV."""
    if model is None:
        model = load_model()
    out = Path(output_dir) if output_dir is not None else DEFAULT_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    baseline = model.baseline
    baseline.to_frame().to_csv(out / "baseline_blocks.csv", index=False)
    pd.DataFrame(
        [
            {
                "population": baseline.total_population,
                "demand": baseline.total_demand,
                "capacity": baseline.total_capacity,
                "demand_within": baseline.total_demand_within,
                "demand_left": baseline.total_demand_left,
                "aggregate_provision": baseline.aggregate_provision,
                "blocks_with_deficit": int((baseline.demand_left > 0).sum()),
            }
        ]
    ).to_csv(out / "baseline_summary.csv", index=False)

    auto = model.auto_search_one_school(capacity)
    auto.ranking.to_csv(out / f"auto_search_{capacity}.csv", index=False)

    # Для примера сценария среди равных лучших берём наиболее населённый
    # квартал; сам рейтинг и список всех лучших от этого не меняются.
    population = model.population
    best_block = max(
        auto.best_block_ids,
        key=lambda bid: (int(population[bid]), -int(bid)),
    )
    manual = model.place_one_school(best_block, capacity)
    manual.to_frame().to_csv(
        out / f"scenario_one_school_{capacity}_best.csv", index=False
    )
    baseline.compare_with(manual).to_csv(
        out / f"scenario_one_school_{capacity}_best_comparison.csv", index=False
    )

    if auto.errors:
        pd.DataFrame(auto.errors).to_csv(
            out / f"auto_search_{capacity}_errors.csv", index=False
        )

    return auto


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    from loguru import logger

    logger.remove()  # не засорять вывод служебными сообщениями BlocksNet

    model = load_model()
    baseline = model.baseline

    print("Исходное состояние:")
    print(f"  население: {baseline.total_population} чел.")
    print(f"  спрос: {baseline.total_demand} мест")
    print(f"  вместимость: {baseline.total_capacity} мест")
    print(f"  покрыто в пределах 15 мин: {baseline.total_demand_within} мест")
    print(f"  остаточный дефицит: {baseline.total_demand_left} мест")
    print(f"  агрегированная обеспеченность: {baseline.aggregate_provision:.6f}")

    auto = write_results(model, capacity=600)

    print()
    print("Автоматический поиск одной школы на 600 мест:")
    print(f"  проверено вариантов: {auto.checked_count} из {auto.attempted_count}")
    print(f"  полный перебор: {'да' if auto.complete else 'нет'}")
    print(f"  лучшие кварталы: {auto.best_block_ids}")
    print(f"  остаточный дефицит: {auto.best_total_deficit} мест")
    print(f"  сокращение дефицита: {auto.best_reduction} мест")

    if auto.errors:
        print("  ошибки:")
        for error in auto.errors:
            print(
                f"    квартал {error['block_id']}: "
                f"{error['error_type']}: {error['message']}"
            )

    print()
    print(f"Результаты сохранены в: {DEFAULT_RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
