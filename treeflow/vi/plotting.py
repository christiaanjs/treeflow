import typing as tp


def _sampled_coordinates(
    name: str,
    n_coords: int,
    tree_vars: tp.Iterable[str],
    coords_per_var: int,
    tree_coords: int,
) -> tp.List[tp.Tuple[str, int]]:
    """Pick a small, representative set of ``(label, coordinate_index)`` pairs.

    - Scalar variables contribute their single coordinate.
    - Variables whose name contains any entry of ``tree_vars`` are treated as
      node-height vectors: the **last** coordinate is taken to be the root and is
      always included, plus up to ``tree_coords - 1`` evenly-spaced internal-node
      coordinates.
    - Any other vector variable contributes ``coords_per_var`` evenly-spaced
      coordinates.
    """
    import numpy as np

    base = name.split(":")[0]
    if n_coords <= 1:
        return [(base, 0)]

    is_tree = any(t in name for t in tree_vars)
    if is_tree:
        root = n_coords - 1
        n_internal = max(min(tree_coords, n_coords) - 1, 0)
        internal = np.unique(
            np.linspace(0, n_coords - 2, n_internal, dtype=int)
        ).tolist() if n_internal else []
        pairs = [(f"{base}[node {i}]", int(i)) for i in internal]
        pairs.append((f"{base}[root]", root))
        return pairs

    k = min(coords_per_var, n_coords)
    idx = np.unique(np.linspace(0, n_coords - 1, k, dtype=int))
    return [(f"{base}[{int(i)}]", int(i)) for i in idx]


def _plot_sampled_traces(parameter_trace, ax, tree_vars, coords_per_var, tree_coords,
                         title):
    """Draw a representative sample of coordinate trajectories into one axis."""
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.5))

    for name, trace in parameter_trace.items():
        arr = np.asarray(trace)
        num_steps = arr.shape[0]
        flat = arr.reshape(num_steps, -1)
        steps = np.arange(num_steps)
        for label, j in _sampled_coordinates(
            name, flat.shape[1], tree_vars, coords_per_var, tree_coords
        ):
            ax.plot(steps, flat[:, j], lw=1.0, label=label)

    ax.set_xlabel("iteration")
    ax.set_ylabel("parameter value")
    ax.legend(fontsize=6, loc="best", ncol=2)
    if title is not None:
        ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    return ax


def plot_parameter_traces(
    parameter_trace: tp.Mapping[str, object],
    sample: bool = False,
    ax=None,
    coords_per_var: int = 1,
    tree_vars: tp.Iterable[str] = ("tree",),
    tree_coords: int = 3,
    title: tp.Optional[str] = None,
    max_individual_lines: int = 16,
    ncols: int = 3,
    axes=None,
    figsize_per_plot: tp.Tuple[float, float] = (4.0, 2.5),
):
    """Plot the optimisation trajectory of each variational parameter.

    This is approximation-agnostic: it works for the mean-field, full-rank and
    IAF families alike. Each entry of ``parameter_trace`` is the trace of one
    ``tf.Variable`` with a leading step axis and an arbitrary trailing shape
    (a scalar for a mean-field location, ``(D, D)`` for a full-rank scale
    matrix, a weight matrix for an IAF layer, ...). Trailing dimensions are
    flattened to coordinates.

    Two layouts are supported:

    - **Full layout** (``sample=False``, the default): one subplot per variable.
      A variable is drawn as one line per coordinate when it has at most
      ``max_individual_lines`` coordinates, or as the per-step mean with a shaded
      min–max envelope plus a handful of sampled coordinate lines otherwise (so
      large full-rank/IAF blocks stay legible).

    - **Sampled layout** (``sample=True``): a small, representative set of
      coordinates — one per scalar variable, ``coords_per_var`` per other vector,
      and the root plus ``tree_coords - 1`` internal-node coordinates for any
      variable named in ``tree_vars`` — is drawn as labelled lines into a single
      axis (``ax``). This is intended for comparing several runs side by side:
      create one subplot per run and pass each ``ax`` in turn. To split a packed
      approximation (e.g. a full-rank ``loc`` vector) into per-variable blocks,
      build ``parameter_trace`` with one named entry per model variable before
      calling this.

    Args:
        parameter_trace: mapping ``name -> trace`` where each trace is array-like
            of shape ``(num_steps, *param_shape)`` (e.g. ``VIResults.parameters``
            returned by ``fit_fixed_topology_variational_approximation``).
        sample: select the sampled single-axis layout instead of one subplot per
            variable.
        ax: single matplotlib ``Axes`` to draw the sampled layout into. A new
            figure/axis is created when omitted. Only used when ``sample=True``.
        coords_per_var: number of evenly-spaced coordinates sampled from each
            (non-tree) vector variable in the sampled layout.
        tree_vars: name substrings identifying node-height vectors; their last
            coordinate is treated as the root.
        tree_coords: total coordinates sampled for a tree variable (root plus
            ``tree_coords - 1`` internal nodes).
        title: optional title for the sampled-layout axis.
        max_individual_lines: per-variable coordinate count above which the full
            layout summarises (mean + min–max envelope) instead of drawing one
            line per coordinate.
        ncols: number of subplot columns in the full layout.
        axes: optional pre-existing flat/2-D array of matplotlib ``Axes`` to draw
            the full layout into. A new figure is created when omitted.
        figsize_per_plot: ``(width, height)`` per subplot for the created figure.

    Returns:
        The single ``Axes`` drawn into when ``sample=True``, otherwise the flat
        array of ``Axes`` for the full layout.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    names = list(parameter_trace.keys())
    n = len(names)
    if n == 0:
        raise ValueError("parameter_trace is empty; nothing to plot.")

    if sample:
        return _plot_sampled_traces(
            parameter_trace, ax, tree_vars, coords_per_var, tree_coords, title
        )

    if axes is None:
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
            squeeze=False,
        )
        created_fig = fig
    else:
        axes = np.asarray(axes)
        created_fig = None
    axes = np.asarray(axes).ravel()

    for ax, name in zip(axes, names):
        trace = np.asarray(parameter_trace[name])
        num_steps = trace.shape[0]
        flat = trace.reshape(num_steps, -1)  # (num_steps, n_coords)
        steps = np.arange(num_steps)
        n_coords = flat.shape[1]

        if n_coords <= max_individual_lines:
            for j in range(n_coords):
                ax.plot(steps, flat[:, j], lw=0.8)
        else:
            ax.fill_between(
                steps, flat.min(axis=1), flat.max(axis=1), alpha=0.2, label="min–max"
            )
            # Overlay a few evenly-spaced coordinates to show actual trajectories.
            sample_idx = np.unique(
                np.linspace(0, n_coords - 1, max_individual_lines, dtype=int)
            )
            for j in sample_idx:
                ax.plot(steps, flat[:, j], lw=0.5, alpha=0.5)
            ax.legend(fontsize=6, loc="best")

        ax.set_title(f"{name}  ({n_coords})", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide any unused axes in the grid.
    for ax in axes[n:]:
        ax.set_visible(False)

    if created_fig is not None:
        created_fig.tight_layout()

    return axes


__all__ = ["plot_parameter_traces"]
