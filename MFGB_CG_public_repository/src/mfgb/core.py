"""Core implementation of Multi-Factor Game-Based (MFGB) flow routing.

The implementation is array based, contains no machine-specific paths, and
accepts precomputed terrain attributes so preprocessing choices remain
explicit. Direction order is E, SE, S, SW, W, NW, N, NE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

DIRS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
)
INVALID = -9999.0


@dataclass
class MFGBParameters:
    """Weights used by the terrain, path, regulation, and curvature terms."""
    slope_weight: float = 0.2
    tpi_weight: float = 0.2
    trd_weight: float = 0.2
    roughness_weight: float = 0.2
    ecv_weight: float = 0.2
    kt_weight: float = 0.2
    curvature_weight: float = 0.2
    congestion_penalty_weight: float = 0.2
    upstream_consistency_weight: float = 0.2

    @classmethod
    def from_dict(cls, params: Optional[Dict[str, float]]) -> "MFGBParameters":
        if params is None:
            return cls()
        defaults = cls()
        values = {name: float(params.get(name, getattr(defaults, name)))
                  for name in cls.__dataclass_fields__}
        return cls(**values)

    def as_dict(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


def compute_tca(flow_directions: np.ndarray, dem: np.ndarray, cell_size: float,
                cells_sorted=None) -> np.ndarray:
    """Compute total contributing area from an 8-direction allocation field."""
    flow_directions = np.asarray(flow_directions, dtype=float)
    dem = np.asarray(dem, dtype=float)
    if dem.ndim != 2:
        raise ValueError("dem must be a two-dimensional array")
    if flow_directions.shape != dem.shape + (8,):
        raise ValueError("flow_directions must have shape (rows, cols, 8)")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    rows, cols = dem.shape
    tca = np.where(dem == INVALID, 0.0, cell_size ** 2).astype(float)

    if cells_sorted is None:
        cells_sorted = sorted(
            [(dem[i, j], i, j) for i in range(rows) for j in range(cols)
             if dem[i, j] != INVALID],
            reverse=True,
        )

    for _, i, j in cells_sorted:
        for d, (di, dj) in enumerate(DIRS):
            ni, nj = i - di, j - dj
            if 0 <= ni < rows and 0 <= nj < cols:
                tca[i, j] += tca[ni, nj] * flow_directions[ni, nj, d]
    return tca


class _FlowAgent:
    def __init__(
        self,
        i: int,
        j: int,
        model: "MFGBModel",
    ) -> None:
        self.i = i
        self.j = j
        self.model = model
        self.weights = np.zeros(8, dtype=float)
        self.prev_weights: Optional[np.ndarray] = None
        self.neighbors = [None] * 8

    def calculate_payoff(self, direction: int) -> float:
        m = self.model
        dem = m.dem
        di, dj = DIRS[direction]
        ni, nj = self.i + di, self.j + dj

        # Invalid or non-downslope directions receive zero allocation.
        if not (0 <= ni < m.rows and 0 <= nj < m.cols):
            return 0.0
        if dem[ni, nj] == INVALID or dem[ni, nj] >= dem[self.i, self.j]:
            return 0.0

        # M1: terrain-informed evidence. Missing optional attributes are
        # skipped; if all attributes are absent, local elevation drop is used.
        terms = (
            (m.params.slope_weight, m.slopes),
            (m.params.tpi_weight, m.tpis),
            (m.params.trd_weight, m.trds),
            (m.params.roughness_weight, m.roughness),
            (m.params.ecv_weight, m.ecvs),
            (m.params.kt_weight, m.kts),
        )
        valid_terms = [
            weight * values[ni, nj]
            for weight, values in terms
            if values[ni, nj] != INVALID and np.isfinite(values[ni, nj])
        ]
        if not valid_terms:
            drop = dem[self.i, self.j] - dem[ni, nj]
            payoff = m.params.slope_weight * drop / (di * di + dj * dj)
        else:
            payoff = float(sum(valid_terms))

        # Neighbor same-direction interaction.
        neighbor = self.neighbors[direction]
        neighbor_weight = neighbor.weights[direction] if neighbor is not None else 0.0
        payoff += neighbor_weight

        # Accumulation-dependent congestion penalty.
        downstream_tca = m.tca[ni, nj]
        congestion = np.tanh(downstream_tca / (m.cell_size ** 2))
        payoff -= m.params.congestion_penalty_weight * congestion
        if direction in (1, 3, 5, 7):
            payoff -= 0.1 * congestion

        # Reverse-direction competition.
        if neighbor is not None:
            opposite = (direction + 4) % 8
            reverse_weight = neighbor.weights[opposite]
            payoff *= (1.0 - 0.7 * reverse_weight)

        # M2: upstream path consistency.
        upstream_consistency = 0
        reverse_dir = (direction + 4) % 8
        di_rev, dj_rev = DIRS[reverse_dir]
        current_i, current_j = self.i, self.j
        for _ in range(m.max_trace_steps):
            ui = current_i + di_rev
            uj = current_j + dj_rev
            if not (0 <= ui < m.rows and 0 <= uj < m.cols):
                break
            if dem[ui, uj] <= dem[current_i, current_j]:
                break
            upstream_agent = m.agents[ui, uj]
            if upstream_agent is None or upstream_agent.weights[direction] <= 0:
                break
            upstream_consistency += 1
            current_i, current_j = ui, uj
        payoff += m.params.upstream_consistency_weight * upstream_consistency

        # Curvature-direction consistency.
        flow_angle = np.arctan2(dj, di)
        curvature_direction = m.curvature_direction[self.i, self.j]
        if curvature_direction == INVALID or not np.isfinite(curvature_direction):
            curvature_direction = 0.0
        angle_diff = abs(curvature_direction - flow_angle)
        curvature_coef = 0.7 * np.cos(angle_diff) + 0.3
        payoff += m.params.curvature_weight * curvature_coef

        # Current research code floors valid-direction payoff at 0.01.
        return max(float(payoff), 0.01)

    def update_strategy(self) -> None:
        self.prev_weights = self.weights.copy()
        scores = np.array([self.calculate_payoff(d) for d in range(8)], dtype=float)
        total = float(scores.sum())
        self.weights = scores / total if total > 0.0 else np.zeros(8, dtype=float)


class MFGBModel:
    """Clean MFGB routing model using precomputed terrain arrays.

    Strategies are updated asynchronously in deterministic raster order.
    Convergence is reached when the maximum directional-weight change is below
    ``tolerance``.
    """

    def __init__(
        self,
        dem: np.ndarray,
        cell_size: float,
        slopes: np.ndarray,
        curvature_direction: np.ndarray,
        tpis: Optional[np.ndarray] = None,
        trds: Optional[np.ndarray] = None,
        roughness: Optional[np.ndarray] = None,
        ecvs: Optional[np.ndarray] = None,
        kts: Optional[np.ndarray] = None,
        params: Optional[Dict[str, float]] = None,
        max_iter: int = 100,
        tolerance: float = 1e-4,
        max_trace_steps: int = 300,
    ) -> None:
        self.dem = np.asarray(dem, dtype=float)
        if self.dem.ndim != 2:
            raise ValueError("dem must be a two-dimensional array")
        self.cell_size = float(cell_size)
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.rows, self.cols = self.dem.shape
        self.slopes = np.asarray(slopes, dtype=float)
        self.curvature_direction = np.asarray(curvature_direction, dtype=float)
        shape = self.dem.shape
        self.tpis = np.full(shape, INVALID) if tpis is None else np.asarray(tpis, dtype=float)
        self.trds = np.full(shape, INVALID) if trds is None else np.asarray(trds, dtype=float)
        self.roughness = np.full(shape, INVALID) if roughness is None else np.asarray(roughness, dtype=float)
        self.ecvs = np.full(shape, INVALID) if ecvs is None else np.asarray(ecvs, dtype=float)
        self.kts = np.full(shape, INVALID) if kts is None else np.asarray(kts, dtype=float)
        self.params = MFGBParameters.from_dict(params)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.max_trace_steps = int(max_trace_steps)

        for name, arr in {
            "slopes": self.slopes,
            "curvature_direction": self.curvature_direction,
            "tpis": self.tpis,
            "trds": self.trds,
            "roughness": self.roughness,
            "ecvs": self.ecvs,
            "kts": self.kts,
        }.items():
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} does not match DEM shape {shape}")

        self.tca = np.where(self.dem == INVALID, 0.0, self.cell_size ** 2).astype(float)
        self.agents = np.empty(shape, dtype=object)
        for i in range(self.rows):
            for j in range(self.cols):
                self.agents[i, j] = None if self.dem[i, j] == INVALID else _FlowAgent(i, j, self)

        for i in range(self.rows):
            for j in range(self.cols):
                agent = self.agents[i, j]
                if agent is None:
                    continue
                neighbors = []
                for di, dj in DIRS:
                    ni, nj = i + di, j + dj
                    neighbors.append(self.agents[ni, nj] if 0 <= ni < self.rows and 0 <= nj < self.cols else None)
                agent.neighbors = neighbors

        self.agent_list = [a for a in self.agents.flat if a is not None]
        self.valid_cells_sorted = sorted(
            [(self.dem[i, j], i, j) for i in range(self.rows) for j in range(self.cols)
             if self.dem[i, j] != INVALID],
            reverse=True,
        )
        self.iterations_run = 0
        self.converged = False

    def get_flow_directions(self) -> np.ndarray:
        flow = np.zeros((self.rows, self.cols, 8), dtype=float)
        for agent in self.agent_list:
            flow[agent.i, agent.j] = agent.weights
        return flow

    def run_iteration(self) -> float:
        # Asynchronous, deterministic raster-order updates.
        max_weight_change = 0.0
        for agent in self.agent_list:
            agent.update_strategy()
            if agent.prev_weights is not None:
                change = float(np.max(np.abs(agent.weights - agent.prev_weights)))
                max_weight_change = max(max_weight_change, change)

        self.tca = compute_tca(
            self.get_flow_directions(), self.dem, self.cell_size, self.valid_cells_sorted
        )
        return max_weight_change

    def fit(self) -> "MFGBModel":
        self.converged = False
        for it in range(self.max_iter):
            max_weight_change = self.run_iteration()
            self.iterations_run = it + 1
            if max_weight_change < self.tolerance:
                self.converged = True
                break
        return self
