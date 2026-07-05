import numpy as np
import hkan.swim_sampling as ss


def edge_isolated_swim_centers(X, y, M, n_vars_out, n_vars_in, n_basis, random_seed=42):
    """
    EDGE-ISOLATED SWIM centers: for each input dimension p, sample M
    candidates using only that column's 1D distance. Then for EACH (q,p)
    edge, independently select n_basis DISTINCT informative pairs and take
    one center per pair (the midpoint).

    **Key features:**
    - Selects n_basis DISTINCT pairs per edge to prevent coverage collapse
    - One center per pair (midpoint) instead of interpolation
    - Each edge gets its own random seed to avoid reusing the same pattern

    Returns
    -------
    ndarray, shape (n_vars_out, n_vars_in, n_basis)

    O(n_vars_in) 
    """
    y = y.reshape(-1, 1)
    centers = np.empty((n_vars_out, n_vars_in, n_basis))
    # For each input dimension p, sample candidates and compute SWIM probabilities
    for p in range(n_vars_in):
        # Use only corresponding dimension p
        # Note: How much does y change relative to this dimension alone
        X_vec = X[:, p].reshape(-1, 1)
        x_a, x_b, y_a, y_b = ss.sample_candidate_pairs(X_vec, y, M, random_seed + p)
        probs = ss.create_swim_probabilities(x_a, x_b, y_a, y_b)
        # For each output neuron q, select n_basis distinct pairs
        for q in range(n_vars_out):
            # Unique seed per (q,p) edge to avoid reusing same selection pattern
            seed_qp = random_seed + q * 10_007 + p
            x_a_sel, x_b_sel, _, _, _ = ss.select_pairs(
                x_a, x_b, y_a, y_b, probs, layer_width=n_basis, random_seed=seed_qp
            )
            # One center per selected pair: use midpoint
            centers[q, p, :] = ((x_a_sel + x_b_sel) / 2).ravel()

    return centers


def neuron_shared_swim_centers(X, y, M, n_vars_out, n_vars_in, n_basis, random_seed=42):
    """
    NEURON-SHARED SWIM centers: one set of candidate pairs sampled per
    output neuron q (using full-D distance). Then n_basis DISTINCT pairs
    are selected per (q,p) edge from that shared candidate pool, one center
    per pair (midpoint).

    **Trade-offs vs edge_isolated:**
    - Cheaper: candidates sampled once per q (vs once per (q,p))
    - Risk of confounding: importance scores computed on full-D distance
    - Same coverage properties: distinct pairs per edge with midpoint centers

    Returns
    -------
    ndarray, shape (n_vars_out, n_vars_in, n_basis)

    O(n_vars_out)
    """
    y = y.reshape(-1, 1)
    centers = np.empty((n_vars_out, n_vars_in, n_basis))

    # For each output neuron q, sample candidate pairs using full-dimensional distance
    for q in range(n_vars_out):
        x_a, x_b, y_a, y_b = ss.sample_candidate_pairs(X, y, M, random_seed + q)
        probs = ss.create_swim_probabilities(x_a, x_b, y_a, y_b)

        # For each input dimension p, select n_basis distinct pairs and use midpoints
        for p in range(n_vars_in):
            # Unique seed per (q,p) edge to avoid reusing same selection pattern
            seed_qp = random_seed + q * 10_007 + p

            x_a_sel, x_b_sel, _, _, _ = ss.select_pairs(
                x_a, x_b, y_a, y_b, probs, layer_width=n_basis, random_seed=seed_qp
            )
            # One center per selected pair: use midpoint of column p
            centers[q, p, :] = ((x_a_sel[:, p] + x_b_sel[:, p]) / 2).ravel()

    return centers


def _pair_dist_to_sigma(x_a_sel, x_b_sel, sigma_scale, sigma_min, sigma_max):
    """
    Convert 1D pair distances to sigma values.
    sigma = sigma_scale / |x_a - x_b|, clipped to [sigma_min, sigma_max].
    """
    pair_dist = np.abs(x_a_sel - x_b_sel).ravel()
    pair_dist = np.clip(pair_dist, a_min=1e-6, a_max=None)
    return np.clip(sigma_scale / pair_dist, sigma_min, sigma_max)


def edge_isolated_swim_sigmas(X, y, M, n_vars_out, n_vars_in, n_basis,
                               random_seed=42, sigma_scale=1.0, dist_percentile=5):
    """
    sigma_min/sigma_max are no longer free parameters you set by hand --
    they're derived directly from the actual distance distribution on each
    dimension, since data is already scaled to [0,1]:
      - max distance in [0,1] is at most 1  -> sigma_min = sigma_scale / 1
      - dist_percentile-th percentile distance -> sigma_max = sigma_scale / floor
    """
    y = y.reshape(-1, 1)
    sigmas = np.empty((n_vars_out, n_vars_in, n_basis))

    for p in range(n_vars_in):
        X_vec = X[:, p].reshape(-1, 1)
        x_a, x_b, y_a, y_b = ss.sample_candidate_pairs(X_vec, y, M, random_seed + p)

        # derive bounds from THIS dimension's own pair-distance distribution
        dists = np.abs(x_a - x_b).ravel()
        floor = np.percentile(dists, dist_percentile)
        sigma_max = sigma_scale / floor
        sigma_min = sigma_scale / dists.max()

        probs = ss.create_swim_probabilities(x_a, x_b, y_a, y_b)

        for q in range(n_vars_out):
            seed_qp = random_seed + q * 10_007 + p
            x_a_sel, x_b_sel, _, _, _ = ss.select_pairs(
                x_a, x_b, y_a, y_b, probs, layer_width=n_basis, random_seed=seed_qp
            )
            sigmas[q, p, :] = _pair_dist_to_sigma(
                x_a_sel, x_b_sel, sigma_scale, sigma_min, sigma_max
            )

    return sigmas


def neuron_shared_swim_sigmas(X, y, M, n_vars_out, n_vars_in, n_basis,
                               random_seed=42, sigma_scale=1.0, dist_percentile=5):
    """
    NEURON-SHARED SWIM sigmas: mirrors neuron_shared_swim_centers exactly
    (same full-D candidate sampling per q, same seeding scheme per (q,p)
    edge), so pairing this with neuron_shared_swim_centers under the SAME
    random_seed gives you sigmas derived from the identical pairs that
    produced the centers.

    sigma_min/sigma_max are no longer free parameters -- they're derived
    directly from each dimension's own pair-distance distribution, same
    approach as edge_isolated_swim_sigmas:
      - max distance in [0,1]-scaled data is at most 1 -> sigma_min = sigma_scale / 1
      - dist_percentile-th percentile distance -> sigma_max = sigma_scale / floor

    Note: distance here is computed on column p only (the projection of the
    full-D pair onto that dimension), matching what the center takes its
    midpoint from -- consistent with neuron_shared_swim_centers, but see the
    docstring caveat there re: full-D pair selection vs per-dimension distance.

    Returns
    -------
    ndarray, shape (n_vars_out, n_vars_in, n_basis)
    """
    y = y.reshape(-1, 1)
    sigmas = np.empty((n_vars_out, n_vars_in, n_basis))

    for q in range(n_vars_out):
        x_a, x_b, y_a, y_b = ss.sample_candidate_pairs(X, y, M, random_seed + q)
        probs = ss.create_swim_probabilities(x_a, x_b, y_a, y_b)

        for p in range(n_vars_in):
            seed_qp = random_seed + q * 10_007 + p
            x_a_sel, x_b_sel, _, _, _ = ss.select_pairs(
                x_a, x_b, y_a, y_b, probs, layer_width=n_basis, random_seed=seed_qp
            )

            # derive bounds from THIS dimension's own pair-distance distribution,
            # projected from the full-D candidate pairs sampled for this neuron q
            # the 5th percentile very close to the smallest value
            dists_p = np.abs(x_a[:, p] - x_b[:, p]).ravel()
            floor = np.percentile(dists_p, dist_percentile)
            sigma_max = sigma_scale / floor
            sigma_min = sigma_scale / dists_p.max()

            sigmas[q, p, :] = _pair_dist_to_sigma(
                x_a_sel[:, p], x_b_sel[:, p], sigma_scale, sigma_min, sigma_max
            )

    return sigmas