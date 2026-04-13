from __future__ import annotations

from collections import deque

import numpy as np
from scipy import ndimage


def ensure_binary(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask, dtype=np.uint8)
    return (array > 0).astype(np.uint8)


def volume_fraction(mask: np.ndarray) -> float:
    binary = ensure_binary(mask)
    return float(binary.mean())


def calculate_smoothness_metric(mask: np.ndarray) -> int:
    binary = ensure_binary(mask).astype(np.int16)
    if binary.size == 0:
        return 0
    horizontal = np.abs(binary[:, :-1] - binary[:, 1:]).sum()
    vertical = np.abs(binary[:-1, :] - binary[1:, :]).sum()
    return int(horizontal + vertical)


def count_islands(mask: np.ndarray) -> int:
    binary = ensure_binary(mask)
    rows, cols = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    islands = 0

    for row in range(rows):
        for col in range(cols):
            if binary[row, col] == 0 or visited[row, col]:
                continue
            islands += 1
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            while queue:
                cur_row, cur_col = queue.popleft()
                for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt_row = cur_row + delta_row
                    nxt_col = cur_col + delta_col
                    if (
                        0 <= nxt_row < rows
                        and 0 <= nxt_col < cols
                        and binary[nxt_row, nxt_col] == 1
                        and not visited[nxt_row, nxt_col]
                    ):
                        visited[nxt_row, nxt_col] = True
                        queue.append((nxt_row, nxt_col))
    return islands


def prune_isolated_cells(mask: np.ndarray, min_neighbors: int = 2) -> np.ndarray:
    binary = ensure_binary(mask)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = ndimage.convolve(binary, kernel, mode="constant", cval=0)
    pruned = np.where((binary == 1) & (neighbors < min_neighbors), 0, binary)
    return pruned.astype(np.uint8)


def frontier_band(mask: np.ndarray, width: int = 2) -> np.ndarray:
    binary = ensure_binary(mask).astype(bool)
    if width < 1:
        return np.zeros_like(binary, dtype=bool)
    structure = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(binary, structure=structure, iterations=width)
    eroded = ndimage.binary_erosion(binary, structure=structure, iterations=width)
    return np.logical_xor(dilated, eroded)


def touches_region(mask: np.ndarray, region_mask: np.ndarray) -> bool:
    binary = ensure_binary(mask).astype(bool)
    region = np.asarray(region_mask, dtype=bool)
    return bool(np.logical_and(binary, region).any())


def retain_components_touching_region(mask: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    binary = ensure_binary(mask)
    region = np.asarray(region_mask, dtype=bool)
    labels, component_count = ndimage.label(binary)
    if component_count == 0:
        return binary

    keep_labels = np.unique(labels[np.logical_and(binary == 1, region)])
    keep_labels = keep_labels[keep_labels > 0]
    if keep_labels.size == 0:
        return np.zeros_like(binary, dtype=np.uint8)
    kept = np.isin(labels, keep_labels)
    return kept.astype(np.uint8)


def composite_penalty(
    mask: np.ndarray,
    *,
    volume_target: float,
    smoothness_weight: float,
    island_weight: float,
    volume_weight: float,
) -> float:
    volume_gap = abs(volume_fraction(mask) - volume_target)
    smoothness = calculate_smoothness_metric(mask) / max(mask.size, 1)
    islands = max(count_islands(mask) - 1, 0)
    return float(
        volume_weight * volume_gap + smoothness_weight * smoothness + island_weight * islands
    )
