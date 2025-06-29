
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_pinn_performance(df, input_features, output_features,
                          train_frac, val_frac,
                          save_path=os.getcwd()):
    """
    Plot PINN performance:
      - one row per (non-time) input feature,
      - then (optionally) target vs time,
      - then prediction vs time,
      - plus (optionally) a separate residual plot with twin y-axis.

    If df["{output}_targ"] contains any NaNs, all target‐related plotting
    (target line, fill_between regions, residuals) is skipped.

    Args:
        df (pd.DataFrame): must contain column 't' for time,
            columns "{output}_targ" and "{output}_pred" for the one output feature,
            and any listed inputs.
        input_features (list of str): e.g. ['t','j','T']  (we ignore 't' when plotting inputs).
        output_features (list of str): exactly one element, e.g. ['u_cell'].
        train_frac (float): fraction of total time used for training.
        val_frac   (float): fraction of total time used for validation.
        save_path (str): directory in which to save "PINN_performance.*" and
                         "Residual_plot.*" (if targets are provided).
    """
    # ensure output dir exists
    os.makedirs(save_path, exist_ok=True)

    # unpack names
    output = output_features[0]
    if output.lower() == "u":
        output = "U_cell"
    targ   = f"{output}_targ".lower()
    pred   = f"{output}_pred".lower()
    tcol = "t"

    # detect missing targets
    targets_provided = df[targ].notna().all()

    # compute time breakpoints
    t_min, t_max = df[tcol].min(), df[tcol].max()
    t_train_max  = t_max * train_frac
    t_val_max    = t_max * (train_frac + val_frac)

    # if we have targets, figure out our y‐limits for the fill regions
    if targets_provided:
        y_lo = min(df[[targ, pred]].min())
        y_hi = max(df[[targ, pred]].max())

    # decide how many rows: inputs + (target?) + prediction
    inputs_to_plot = [f for f in input_features if f != tcol]
    n_inputs       = len(inputs_to_plot)
    n_rows         = n_inputs + 1 + (1 if targets_provided else 0)

    fig = plt.figure(figsize=(10, 3 * n_rows), constrained_layout=True)
    gs  = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.0)

    # 1) plot each input feature
    for i, feat in enumerate(inputs_to_plot):
        ax = fig.add_subplot(gs[i])
        ax.plot(df[tcol], df[feat], linewidth=2)
        ax.set_ylabel(feat)
        ax.label_outer()  # hide x‐labels except on the bottom

    row = n_inputs

    # 2) optionally plot the true target + train/val bands
    if targets_provided:
        ax_t = fig.add_subplot(gs[row], sharex=(fig.axes[0] if n_inputs else None))
        ax_t.plot(df[tcol], df[targ],
                  linewidth=3, color='xkcd:cobalt', label='target')
        ax_t.fill_between([t_min, t_train_max], [y_lo]*2, [y_hi]*2,
                          color='green',   alpha=0.25, label='training')
        ax_t.fill_between([t_train_max, t_val_max], [y_lo]*2, [y_hi]*2,
                          color='orange', alpha=0.25, label='validation')
        ax_t.set_ylabel(f"Target {output}")
        ax_t.legend(loc='upper right', frameon=False)
        ax_t.label_outer()
        row += 1

    # 3) always plot the prediction
    ax_p = fig.add_subplot(gs[row], sharex=(ax_t if targets_provided else (fig.axes[0] if n_inputs else None)))
    ax_p.plot(df[tcol], df[pred],
              linewidth=3, color='xkcd:forest green', label='prediction')

    if targets_provided:
        # repeat the same fill regions behind the prediction
        ax_p.fill_between([t_min, t_train_max], [y_lo]*2, [y_hi]*2,
                          color='green',   alpha=0.25)
        ax_p.fill_between([t_train_max, t_val_max], [y_lo]*2, [y_hi]*2,
                          color='orange', alpha=0.25)
        ax_p.legend(loc='upper right', frameon=False)
    else:
        ax_p.legend(loc='upper right', frameon=False)

    ax_p.set_ylabel(f"Predicted {output}")
    ax_p.set_xlabel("Time")

    # save performance figure
    perf_png = os.path.join(save_path, "PINN_performance.png")
    perf_pdf = os.path.join(save_path, "PINN_performance.pdf")
    fig.savefig(perf_png, dpi=300, transparent=True)
    fig.savefig(perf_pdf, format="pdf")
    plt.close(fig)

    # 4) if we have targets, also plot residuals
    if targets_provided:
        resid     = df[targ] - df[pred]
        rel_resid = resid / df[targ] * 100

        fig2, ax1 = plt.subplots(figsize=(10, 3 * n_rows), constrained_layout=True)
        ax1.plot(df[tcol], resid,        alpha=0.6, label='Residual', color = "xkcd:cobalt")
        ax1.set_xlabel("Time")
        ax1.set_ylabel(f"Residual ({targ} − {pred})")

        ax2 = ax1.twinx()
        ax2.plot(df[tcol], rel_resid, linestyle='--', alpha=0.6, label='Rel. Resid (%)', color = "xkcd:brownish red")
        ax2.set_ylabel("Relative Residual (%)")

        # combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right', frameon=False)

        resid_png = os.path.join(save_path, "Residual_plot.png")
        resid_pdf = os.path.join(save_path, "Residual_plot.pdf")
        fig2.savefig(resid_png, dpi=300, transparent=True)
        fig2.savefig(resid_pdf, format="pdf")
        plt.close(fig2)

    print(f"✅ PINN performance plot saved to {save_path}"
          + ("" if not targets_provided else " (including residuals)"))
