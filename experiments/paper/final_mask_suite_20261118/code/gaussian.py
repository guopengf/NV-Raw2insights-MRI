"""Gaussian weights and sampler reused from the project k-t Gaussian implementation."""
import numpy as np

def create_gaussian_weight_matrix(mask_size: tuple[int, int], sigma_x: float, sigma_y: float) -> np.ndarray:
    width, height = mask_size
    x = np.linspace(1 - (width + 1) / 2.0, width - (width + 1) / 2.0, width)
    y = np.linspace(1 - (height + 1) / 2.0, height - (height + 1) / 2.0, height)
    x_grid, y_grid = np.meshgrid(x, y)
    return np.exp(-(x_grid**2 / (2.0 * sigma_x**2) + y_grid**2 / (2.0 * sigma_y**2)))


def random_sampling_optimized(
    mask_size: tuple[int, int],
    total_points: int,
    weight: np.ndarray,
    min_dist_lookup: np.ndarray,
    rng: np.random.Generator,
    existing_mask: np.ndarray,
) -> np.ndarray:
    if total_points <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    width, height = mask_size
    sampled_points: list[list[int]] = []
    current_weight = weight * (1 - existing_mask.astype(np.float32))
    if np.sum(current_weight) <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    forbidden_mask = np.zeros((height, width), dtype=bool)
    batch_size = max(total_points * 2, 1000)
    prob = current_weight.ravel() / np.sum(current_weight)
    indices = rng.choice(width * height, size=batch_size, p=prob)

    count = 0
    idx_ptr = 0
    y_grid, x_grid = np.ogrid[:height, :width]

    while count < total_points and idx_ptr < batch_size:
        idx = int(indices[idx_ptr])
        idx_ptr += 1
        y, x = divmod(idx, width)

        if forbidden_mask[y, x] or existing_mask[y, x]:
            continue

        sampled_points.append([x + 1, y + 1])
        count += 1

        dist = float(min_dist_lookup[y, x])
        if dist > 0:
            y_min = max(0, int(y - dist))
            y_max = min(height, int(y + dist + 1))
            x_min = max(0, int(x - dist))
            x_max = min(width, int(x + dist + 1))
            region_y = y_grid[y_min:y_max, 0]
            region_x = x_grid[0, x_min:x_max]
            dist_sq = (region_y[:, None] - y) ** 2 + (region_x - x) ** 2
            forbidden_mask[y_min:y_max, x_min:x_max] |= dist_sq < dist**2

        if idx_ptr >= batch_size and count < total_points:
            current_weight = weight * (1 - existing_mask.astype(np.float32)) * (1 - forbidden_mask)
            if np.sum(current_weight) <= 0:
                break
            prob = current_weight.ravel() / np.sum(current_weight)
            indices = rng.choice(width * height, size=batch_size, p=prob)
            idx_ptr = 0

    if not sampled_points:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sampled_points, dtype=np.int64)


def generate_pattern(shape, counts, seed, acs):
    """Condition the project Gaussian sampler on an explicit ACS and exact counts."""
    nz, ny = shape
    rng = np.random.default_rng(seed)
    initial = create_gaussian_weight_matrix((ny, nz), ny/5, nz/5)
    distance = 3 * ((1 - initial) / 2 + .5)
    weights = initial.copy()
    masks = np.empty((len(counts), nz, ny), np.uint8)
    fill_counts = []
    for t, count in enumerate(counts):
        mask = acs.copy()
        for _ in range(2):
            missing = int(count - mask.sum())
            xy = random_sampling_optimized((ny, nz), missing, weights, distance, rng, mask)
            if xy.size:
                mask[xy[:, 1]-1, xy[:, 0]-1] = True
        missing = int(count - mask.sum())
        if missing:
            allowed = np.flatnonzero(~mask)
            prob = weights.ravel()[allowed]
            mask.flat[rng.choice(allowed, missing, replace=False, p=prob/prob.sum())] = True
        weights[mask & ~acs] *= .5
        # Prevent underflow in very long synthetic studies.
        np.maximum(weights, np.finfo(float).tiny, out=weights)
        masks[t] = mask
        fill_counts.append(missing)
    return masks, {'gaussian_weighted_fill_outside_distance_passes': fill_counts}


