import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def augment_history_features(df, window_current='1h'):
    """
    Augment dataset with historical features to highlight historical and usage of the dataset.

    Required input columns:
      - 't': time (hours)
      - 'j': current density (A/cm²)

    Any other columns (e.g., 'U') are preserved untouched.

    Adds:
      - cumulative_current: ∫ j dt (A·h/cm²)
      - rolling_mean_current: rolling mean of j over window_current
      - rolling_std_current: rolling std of j over window_current
    """

    # 1) Check required columns
    required = {'t', 'j'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    # 2) Sort & reset so we have a clean integer index
    df = df.copy().sort_values('t').reset_index(drop=True)

    # 3) Compute raw dt (in hours) on the integer index
    dt = df['t'].diff().fillna(0)

    # 4) Cumulative current—this now lines up one-to-one with j & dt
    df['cumulative_current'] = (df['j'] * dt).cumsum()

    # 5) Build a DateTimeIndex used only for rolling window calculations
    df['time_delta'] = pd.to_timedelta(df['t'], unit='h')
    df = df.set_index('time_delta')

    # 6) Rolling mean and std on j over the specified window
    df['rolling_mean_current'] = (
        df['j']
          .rolling(window=window_current, min_periods=1)
          .mean()
    )
    df['rolling_std_current']  = (
        df['j']
          .rolling(window=window_current, min_periods=1)
          .std()
          .fillna(0)
    )

    # 8) Clean up and return
    df = df.reset_index(drop=True)
    return df


def downsample_df(df: pd.DataFrame, max_points: int = 200_000) -> pd.DataFrame:
    """
    Randomly downsample to max_points rows, preserving sort order on 't'.
    """
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42).sort_values('t')
    return df



def plot_feature_dashboard(df: pd.DataFrame, max_points: int = 200_000):
    """
    Interactive Plotly dashboard for original ('t','j') and augmented features.
    Ignores extra columns; uses Scattergl for performance with large data.
    """
    # 1) Downsample if needed
    df_plot = downsample_df(df, max_points)

    # 2) Define which columns to plot
    originals = ['j']
    aug_feats = [
        c for c in [
            'cumulative_current',
            'rolling_mean_current',
            'rolling_std_current',
        ]
        if c in df_plot.columns
    ]

    # 3) Build a 2×C grid where C is max(#originals, #aug_feats)
    n_cols = max(len(originals), len(aug_feats))
    n_rows = 2

    # 4) Create a flat list of titles, padding each row to length n_cols
    top_titles    = originals + ['']*(n_cols - len(originals))
    bottom_titles = aug_feats  + ['']*(n_cols - len(aug_feats))
    all_titles    = top_titles + bottom_titles

    fig = make_subplots(
        rows=2, cols=n_cols,
        subplot_titles=all_titles,
        shared_xaxes=True,
        vertical_spacing=0.12,      # a bit more room between rows
    )

    # 5) Add the original-feature traces in row 1
    for i, feat in enumerate(originals, start=1):
        fig.add_trace(
            go.Scattergl(
                x=df_plot['t'],
                y=df_plot[feat],
                mode='markers',
                marker=dict(size=2, opacity=0.4),
                name=feat
            ),
            row=1, col=i
        )

    # 6) Add the augmented‐feature traces in row 2
    for i, feat in enumerate(aug_feats, start=1):
        fig.add_trace(
            go.Scattergl(
                x=df_plot['t'],
                y=df_plot[feat],
                mode='markers',
                marker=dict(size=2, opacity=0.4),
                name=feat
            ),
            row=2, col=i
        )

    # 7) Put the x-axis label only on the bottom row
    for col in range(1, n_cols+1):
        fig.update_xaxes(
            title_text="t (hours)",
            row=2, col=col
        )

    # 8) Layout tweaks
    fig.update_layout(
        title_text="Original vs. Augmented Features Over Time",
        height=600,
        width=300 * n_cols,
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    fig.show()




# Example usage:
# df = pd.read_csv('datasets/normalized_solarPV_data_for_evaluation.csv')  # can contain 't', 'j', plus others like 'U'
# aug_df = augment_history_features(df, window_current='2h')
# aug_df.to_csv('datasets/augmented_HPRO.csv', index = False )

# plot_feature_dashboard(aug_df, max_points=150_000)
