from __future__ import annotations

import numpy as np


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix)]


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    right_t = transpose(right)
    return [[sum(x * y for x, y in zip(row, column)) for column in right_t] for row in left]


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(value * coefficient for value, coefficient in zip(row, vector)) for row in matrix]


def matrix_rank(matrix: list[list[float]], tolerance: float | None = None) -> int:
    """Scale columns before applying a relative SVD rank test.

    X, X'WX and the KKT matrix live on very different numeric scales.  A fixed
    absolute tolerance therefore gives mutually contradictory ranks even though
    column scaling cannot change mathematical rank.
    """
    if not matrix:
        return 0
    values = np.asarray(matrix, dtype=float)
    scales = np.linalg.norm(values, axis=0)
    nonzero = scales > 0
    if not np.any(nonzero):
        return 0
    scaled = values[:, nonzero] / scales[nonzero]
    return int(np.linalg.matrix_rank(scaled, tol=tolerance))


def solve(matrix: list[list[float]], vector: list[float], tolerance: float = 1e-12) -> list[float]:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix) or len(vector) != size:
        raise ValueError("LINEAR_SYSTEM_SHAPE_INVALID")
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= tolerance:
            raise ValueError("LINEAR_SYSTEM_SINGULAR")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            multiple = augmented[row][column]
            augmented[row] = [
                value - multiple * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [row[-1] for row in augmented]


def inverse(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    columns = []
    for column in range(size):
        unit = [1.0 if row == column else 0.0 for row in range(size)]
        columns.append(solve(matrix, unit))
    return transpose(columns)


def infinity_norm(matrix: list[list[float]]) -> float:
    return max((sum(abs(value) for value in row) for row in matrix), default=0.0)


def condition_number(matrix: list[list[float]]) -> float:
    if not matrix:
        return float("inf")
    value = float(np.linalg.cond(np.asarray(matrix, dtype=float)))
    return value if np.isfinite(value) else float("inf")


def null_space_basis(matrix: list[list[float]], columns: int, tolerance: float | None = None) -> list[list[float]]:
    """Return orthonormal null(C) basis vectors as columns of a row-major matrix."""
    if not matrix:
        return [[1.0 if row == column else 0.0 for column in range(columns)] for row in range(columns)]
    if any(len(row) != columns for row in matrix):
        raise ValueError("NULL_SPACE_SHAPE_INVALID")
    constraints = np.asarray(matrix, dtype=float)
    _, singular_values, vh = np.linalg.svd(constraints, full_matrices=True)
    rank = int(np.linalg.matrix_rank(constraints, tol=tolerance))
    return vh[rank:, :].T.tolist()


def weighted_normal_equations(
    exposures: list[list[float]], targets: list[float], weights: list[float],
) -> tuple[list[list[float]], list[float]]:
    factor_count = len(exposures[0])
    gram = [[0.0] * factor_count for _ in range(factor_count)]
    rhs = [0.0] * factor_count
    for row, target, weight in zip(exposures, targets, weights):
        for left in range(factor_count):
            rhs[left] += weight * row[left] * target
            for right in range(factor_count):
                gram[left][right] += weight * row[left] * row[right]
    return gram, rhs


def constrained_weighted_least_squares(
    exposures: list[list[float]],
    targets: list[float],
    weights: list[float],
    constraints: list[list[float]],
) -> tuple[list[float], float, int]:
    if not exposures or len(exposures) != len(targets) or len(targets) != len(weights):
        raise ValueError("WLS_ROW_AXIS_INVALID")
    x, _, basis_array, exposure_rank = prepare_constrained_design(
        exposures, constraints
    )
    coefficients, numeric_condition = solve_prepared_weighted_least_squares(
        x, np.asarray(targets, dtype=float), np.asarray(weights, dtype=float),
        basis_array,
    )
    return coefficients.tolist(), numeric_condition, exposure_rank


def prepare_constrained_design(
    exposures: list[list[float]], constraints: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate time-invariant identification once per daily cross-section."""
    if not exposures:
        raise ValueError("WLS_ROW_AXIS_INVALID")
    factor_count = len(exposures[0])
    if factor_count == 0 or any(len(row) != factor_count for row in exposures):
        raise ValueError("WLS_DESIGN_INVALID")
    if any(len(row) != factor_count for row in constraints):
        raise ValueError("WLS_CONSTRAINT_SHAPE_INVALID")
    x = np.asarray(exposures, dtype=float)
    c = np.asarray(constraints, dtype=float) if constraints else np.empty((0, factor_count))
    gram_array = x.T @ x
    gram = gram_array.tolist()
    exposure_rank = matrix_rank(exposures)
    constraint_rank = matrix_rank(constraints) if constraints else 0
    expected_rank = factor_count - constraint_rank
    if exposure_rank != expected_rank:
        raise ValueError(
            f"WLS_UNEXPECTED_RANK_DEFICIENCY rank={exposure_rank} expected={expected_rank}"
        )
    stacked_rank = matrix_rank(np.vstack([x, c]).tolist())
    if stacked_rank != factor_count:
        raise ValueError(
            f"WLS_CONSTRAINT_IDENTIFICATION_FAILED rank={stacked_rank} columns={factor_count}"
        )
    constraint_count = len(constraints)
    if constraints:
        kkt = [
            gram[row] + [constraints[column][row] for column in range(constraint_count)]
            for row in range(factor_count)
        ] + [constraint[:] + [0.0] * constraint_count for constraint in constraints]
        if matrix_rank(kkt) != factor_count + constraint_count:
            raise ValueError("WLS_KKT_SYSTEM_RANK_DEFICIENT")
    basis_array = np.asarray(null_space_basis(constraints, factor_count), dtype=float)
    return x, c, basis_array, exposure_rank


def solve_prepared_weighted_least_squares(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, basis_array: np.ndarray,
    constraint_array: np.ndarray | None = None,
    *, compute_condition: bool = True,
) -> tuple[np.ndarray, float]:
    """Solve one weight iteration in a scale-invariant canonical factor basis."""
    if x.shape[0] != y.shape[0] or y.shape[0] != w.shape[0]:
        raise ValueError("WLS_ROW_AXIS_INVALID")
    if constraint_array is None:
        reduced_x = x @ basis_array
        coefficient_map = basis_array
    else:
        if constraint_array.ndim != 2 or constraint_array.shape[1] != x.shape[1]:
            raise ValueError("WLS_CONSTRAINT_SHAPE_INVALID")
        column_scales = np.sqrt(np.sum(w[:, None] * np.square(x), axis=0))
        if np.any(~np.isfinite(column_scales)) or np.any(column_scales <= 0):
            raise ValueError("WLS_COLUMN_SCALE_INVALID")
        canonical_x = x / column_scales[None, :]
        canonical_constraints = constraint_array / column_scales[None, :]
        canonical_basis = np.asarray(
            null_space_basis(canonical_constraints.tolist(), x.shape[1]), dtype=float
        )
        reduced_x = canonical_x @ canonical_basis
        coefficient_map = canonical_basis / column_scales[:, None]
    reduced_gram = reduced_x.T @ (w[:, None] * reduced_x)
    reduced_rhs = reduced_x.T @ (w * y)
    singular_gram = False
    try:
        reduced_coefficients = np.linalg.solve(reduced_gram, reduced_rhs)
    except np.linalg.LinAlgError as exc:
        # Forming X'WX squares the condition number: a full-rank design may
        # have a numerically singular Gram. Recover diagnostic coefficients
        # from the weighted design, without ridge or relaxing identification.
        # Such a result must remain invalid at the existing condition gate.
        weighted_x = np.sqrt(w)[:, None] * reduced_x
        reduced_coefficients, _, rank, _ = np.linalg.lstsq(
            weighted_x, np.sqrt(w) * y, rcond=None,
        )
        if rank != reduced_x.shape[1]:
            raise ValueError("LINEAR_SYSTEM_SINGULAR") from exc
        singular_gram = True
    coefficients = coefficient_map @ reduced_coefficients
    # The condition number is a diagnostic, not part of the coefficient solve.
    # IRLS may call this function many times with the same daily design; callers
    # can defer the SVD-backed diagnostic until the final robust solution.
    numeric_condition = (
        float(np.linalg.cond(reduced_gram)) if compute_condition else float("nan")
    )
    if singular_gram:
        numeric_condition = float("inf")
    return coefficients, numeric_condition
