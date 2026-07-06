import numpy as np
import hkan.swim_sampling as ss


def _pair_dist_to_sigma(x_a_sel, x_b_sel, sigma_scale, sigma_min, sigma_max):
    """
    Convert 1D pair distances to sigma values.
    sigma = sigma_scale / |x_a - x_b|, clipped to [sigma_min, sigma_max].
    """
    pair_dist = np.abs(x_a_sel - x_b_sel).ravel()
    pair_dist = np.clip(pair_dist, a_min=1e-6, a_max=None)
    return np.clip(sigma_scale / pair_dist, sigma_min, sigma_max)


def edge_isolated_swim(X, y, M, n_vars_out, n_vars_in, n_basis, random_seed=42,
                        use_swim_sigma=True, sigma_scale=1.0, dist_percentile=5,
                        fixed_sigma=1.0):
    """
    EDGE-ISOLATED SWIM centers + sigmas, computed together from the SAME
    selected pairs -- merges what used to be edge_isolated_swim_centers and
    edge_isolated_swim_sigmas into one function so centers/sigmas can never
    desynchronize (no separate seed/M bookkeeping required to keep them paired).

    For each input dimension p, sample M candidates using only that column's
    1D distance. Then for EACH (q,p) edge, independently select n_basis
    DISTINCT informative pairs; take one center per pair (the midpoint), and
    optionally one sigma per pair (inverse pair-distance, SWIM-style).

    Parameters
    ----------
    X, y, M : training data and candidate-pair count, as in the original functions.
    n_vars_out, n_vars_in, n_basis : layer shape.
    random_seed : shared seed driving both centers and sigmas (no longer two
        separate seeds to keep in sync -- there's only one call now).
    use_swim_sigma : bool
        If True (default): sigma is derived from each selected pair's own
        distance (sigma = sigma_scale / |x_a - x_b|, bounds auto-derived from
        this dimension's pair-distance distribution -- see sigma_scale note
        below). If False: sigma is set to the constant `fixed_sigma` for
        every basis function in every block, i.e. plain fixed-sigma behavior,
        while centers are still SWIM-selected.
    sigma_scale : float
        Only used when use_swim_sigma=True. Scale factor in
        sigma = sigma_scale / |x_a - x_b|. Uncalibrated by default -- compare
        the resulting median sigma against Table IV's tuned values for the
        function/layer you're targeting before assuming placement quality
        is the bottleneck.
    dist_percentile : float
        Only used when use_swim_sigma=True. Percentile used both as the
        distance floor inside SWIM's probability computation (matching
        create_swim_probabilities) and to derive this dimension's sigma_max
        (sigma_max = sigma_scale / floor), so the two stay mechanically
        consistent instead of being picked independently.
    fixed_sigma : float
        Only used when use_swim_sigma=False. Constant sigma value assigned
        to every basis function.

    Returns
    -------
    centers : ndarray, shape (n_vars_out, n_vars_in, n_basis)
    sigmas  : ndarray, shape (n_vars_out, n_vars_in, n_basis)

    O(n_vars_in) SWIM calls total (same complexity as the original centers fn).
    """
    y = y.reshape(-1, 1)
    centers = np.empty((n_vars_out, n_vars_in, n_basis))
    sigmas = np.empty((n_vars_out, n_vars_in, n_basis))

    for p in range(n_vars_in):
        # Use only corresponding dimension p
        # Note: how much does y change relative to this dimension alone
        X_vec = X[:, p].reshape(-1, 1)
        x_a, x_b, y_a, y_b = ss.sample_candidate_pairs(X_vec, y, M, random_seed + p)
        probs = ss.create_swim_probabilities(x_a, x_b, y_a, y_b)

        if use_swim_sigma:
            # derive bounds from THIS dimension's own pair-distance distribution,
            # anchored to the same dist_percentile used for the probability floor
            dists = np.abs(x_a - x_b).ravel()
            floor = np.percentile(dists, dist_percentile)
            sigma_max = sigma_scale / floor
            sigma_min = sigma_scale / dists.max()

        # For each output neuron q, select n_basis distinct pairs
        for q in range(n_vars_out):
            # Unique seed per (q,p) edge to avoid reusing same selection pattern
            seed_qp = random_seed + q * 10_007 + p
            x_a_sel, x_b_sel, _, _, _ = ss.select_pairs(
                x_a, x_b, y_a, y_b, probs, layer_width=n_basis, random_seed=seed_qp
            )
            # One center per selected pair: use midpoint
            centers[q, p, :] = ((x_a_sel + x_b_sel) / 2).ravel()

            if use_swim_sigma:
                # sigma derived from the SAME pairs that produced the centers above --
                # guaranteed consistent, since both come from x_a_sel/x_b_sel directly
                sigmas[q, p, :] = _pair_dist_to_sigma(
                    x_a_sel, x_b_sel, sigma_scale, sigma_min, sigma_max
                )
            else:
                # fixed sigma path: centers still SWIM-selected, sharpness is constant
                sigmas[q, p, :] = fixed_sigma

    return centers, sigmas


def neuron_shared_swim(X, y, M, n_vars_out, n_vars_in, n_basis, random_seed=42,
                        use_swim_sigma=True, sigma_scale=1.0, dist_percentile=5,
                        fixed_sigma=1.0):
    """
    NEURON-SHARED SWIM centers + sigmas, computed together from the SAME
    selected pairs -- merges what used to be neuron_shared_swim_centers and
    neuron_shared_swim_sigmas into one function, same rationale as
    edge_isolated_swim above.

    One set of candidate pairs is sampled per output neuron q (using full-D
    distance). Then n_basis DISTINCT pairs are selected per (q,p) edge from
    that shared candidate pool; one center per pair (midpoint of column p),
    and optionally one sigma per pair (inverse distance on column p).

    Trade-offs vs edge_isolated (unchanged from the original docstrings):
    - Cheaper: candidates sampled once per q (vs once per (q,p))
    - Risk of confounding: importance scores computed on full-D distance,
      which also grows with sqrt(n_vars_in) -- consider normalizing distance
      by sqrt(D) inside create_swim_probabilities if comparing across
      datasets with very different input dimensionality (e.g. TF4's 10D).

    Parameters
    ----------
    Same as edge_isolated_swim above, with the same use_swim_sigma /
    sigma_scale / dist_percentile / fixed_sigma semantics. The only
    difference: distances (for both scoring and sigma) are computed on the
    full-D candidate pool, then projected onto column p for centers/sigmas,
    rather than resampling per-dimension.

    Returns
    -------
    centers : ndarray, shape (n_vars_out, n_vars_in, n_basis)
    sigmas  : ndarray, shape (n_vars_out, n_vars_in, n_basis)

    O(n_vars_out) SWIM calls total (same complexity as the original centers fn).
    """
    y = y.reshape(-1, 1)
    centers = np.empty((n_vars_out, n_vars_in, n_basis))
    sigmas = np.empty((n_vars_out, n_vars_in, n_basis))

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

            if use_swim_sigma:
                # bounds derived from THIS dimension's own distances, projected from
                # the full-D candidate pool sampled for this neuron q
                dists_p = np.abs(x_a[:, p] - x_b[:, p]).ravel()
                floor = np.percentile(dists_p, dist_percentile)
                sigma_max = sigma_scale / floor
                sigma_min = sigma_scale / dists_p.max()

                # sigma derived from the SAME pairs used for the center above
                sigmas[q, p, :] = _pair_dist_to_sigma(
                    x_a_sel[:, p], x_b_sel[:, p], sigma_scale, sigma_min, sigma_max
                )
            else:
                # fixed sigma path: centers still SWIM-selected, sharpness is constant
                sigmas[q, p, :] = fixed_sigma

    return centers, sigmas