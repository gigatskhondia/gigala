from __future__ import annotations

from typing import Iterable

import numpy as np

from .metrics import ensure_binary, frontier_band


def infer_stage_resolutions(final_resolution: int) -> tuple[int, ...]:
    if final_resolution <= 16:
        return (final_resolution,)
    if final_resolution <= 32:
        return (16, final_resolution)
    return (16, 32, final_resolution)


def upsample_binary_mask(mask: np.ndarray, target_resolution: int) -> np.ndarray:
    binary = ensure_binary(mask)
    if binary.shape[0] != binary.shape[1]:
        raise ValueError("Only square masks are supported.")
    source = binary.shape[0]
    if source == target_resolution:
        return binary.copy()
    factor = int(np.ceil(target_resolution / source))
    upsampled = np.repeat(np.repeat(binary, factor, axis=0), factor, axis=1)
    return ensure_binary(upsampled[:target_resolution, :target_resolution])


def downsample_binary_mask(mask: np.ndarray, target_resolution: int) -> np.ndarray:
    binary = ensure_binary(mask)
    if binary.shape[0] == target_resolution:
        return binary.copy()
    if binary.shape[0] < target_resolution:
        return upsample_binary_mask(binary, target_resolution)
    factor = binary.shape[0] / target_resolution
    result = np.zeros((target_resolution, target_resolution), dtype=np.uint8)
    for row in range(target_resolution):
        for col in range(target_resolution):
            row_start = int(round(row * factor))
            row_end = int(round((row + 1) * factor))
            col_start = int(round(col * factor))
            col_end = int(round((col + 1) * factor))
            patch = binary[row_start:row_end, col_start:col_end]
            result[row, col] = 1 if patch.mean() >= 0.5 else 0
    return result


def stack_observation(
    mask: np.ndarray,
    support_load_mask: np.ndarray,
    frontier_width: int = 2,
) -> np.ndarray:
    binary = ensure_binary(mask).astype(np.float32)
    support = np.asarray(support_load_mask, dtype=np.float32)
    frontier = frontier_band(binary, width=frontier_width).astype(np.float32)
    return np.stack([binary, support, frontier], axis=0)


def boundary_patch_coordinates(mask: np.ndarray, patch_size: int, width: int = 2) -> list[tuple[int, int]]:
    binary = ensure_binary(mask)
    frontier = frontier_band(binary, width=width)
    limit = binary.shape[0] - patch_size + 1
    coordinates: list[tuple[int, int]] = []
    for row in range(limit):
        for col in range(limit):
            patch_frontier = frontier[row : row + patch_size, col : col + patch_size]
            if patch_frontier.any():
                coordinates.append((row, col))
    return coordinates


def apply_patch(mask: np.ndarray, row: int, col: int, patch_size: int, value: int) -> np.ndarray:
    binary = ensure_binary(mask).copy()
    binary[row : row + patch_size, col : col + patch_size] = 1 if value else 0
    return binary


def immutable_element_mask(resolution: int) -> np.ndarray:
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    mask[0, 0] = 1
    mask[0, -1] = 1
    mask[-1, -1] = 1
    return mask


def choose_best_masks(candidates: Iterable[tuple[np.ndarray, float]], count: int) -> list[np.ndarray]:
    ranked = sorted(candidates, key=lambda item: item[1])
    return [ensure_binary(mask) for mask, _score in ranked[:count]]
