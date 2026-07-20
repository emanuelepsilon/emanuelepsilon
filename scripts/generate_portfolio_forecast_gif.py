from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ASSETS = ["Fund A", "Fund B", "Stock X", "Stock Y", "Stock Z"]
WEIGHTS = np.array([0.15, 0.10, 0.20, 0.25, 0.30])
RETURNS = np.array([0.06, 0.07, 0.10, 0.12, 0.15])
VOLATILITY = np.array([0.10, 0.12, 0.20, 0.25, 0.30])
CORRELATION = np.array(
    [
        [1.00, 0.85, 0.60, 0.55, 0.50],
        [0.85, 1.00, 0.58, 0.53, 0.48],
        [0.60, 0.58, 1.00, 0.75, 0.65],
        [0.55, 0.53, 0.75, 1.00, 0.70],
        [0.50, 0.48, 0.65, 0.70, 1.00],
    ]
)


def simulate_portfolios(seed=42, simulations=500, years=10, steps_per_year=52):
    rng = np.random.default_rng(seed)
    steps = years * steps_per_year
    dt = 1 / steps_per_year
    covariance = np.outer(VOLATILITY, VOLATILITY) * CORRELATION
    shocks = rng.multivariate_normal(
        np.zeros(len(ASSETS)), covariance * dt, size=(simulations, steps)
    )
    drift = (RETURNS - 0.5 * VOLATILITY**2) * dt
    log_returns = drift + shocks
    relative_prices = np.exp(
        np.concatenate(
            [
                np.zeros((simulations, 1, len(ASSETS))),
                np.cumsum(log_returns, axis=1),
            ],
            axis=1,
        )
    )
    paths = relative_prices @ WEIGHTS * 1_000_000
    timeline = np.linspace(0, years, steps + 1)
    return timeline, paths


def make_animation(output_path):
    timeline, paths = simulate_portfolios()
    low, median, high = np.percentile(paths, [10, 50, 90], axis=0)
    mean = paths.mean(axis=0)
    sample_paths = paths[:45]
    ymax = np.percentile(paths[:, -1], 98) * 1.08
    frame_indices = np.linspace(1, len(timeline) - 1, 94, dtype=int)
    frame_indices = np.concatenate([frame_indices, np.repeat(frame_indices[-1], 12)])

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=100)
    fig.patch.set_facecolor("#05070a")

    def draw(frame_number):
        end = frame_indices[frame_number] + 1
        ax.clear()
        ax.set_facecolor("#05070a")
        for path in sample_paths:
            ax.plot(
                timeline[:end], path[:end], color="#2563eb", alpha=0.12, linewidth=0.7
            )
        ax.fill_between(
            timeline[:end], low[:end], high[:end], color="#38bdf8", alpha=0.16
        )
        ax.plot(
            timeline[:end], median[:end], color="#67e8f9", linewidth=1.7, label="Median"
        )
        ax.plot(
            timeline[:end], mean[:end], color="#facc15", linewidth=2.2, label="Expected value"
        )
        ax.axhline(1_000_000, color="#94a3b8", linestyle="--", linewidth=0.9, alpha=0.7)

        current = end - 1
        probability_above_start = np.mean(paths[:, current] > 1_000_000) * 100
        metrics = (
            f"Year {timeline[current]:.1f}\n"
            f"Expected  ${mean[current] / 1_000_000:.2f}M\n"
            f"P(value > start)  {probability_above_start:.0f}%"
        )
        ax.text(
            0.025,
            0.95,
            metrics,
            transform=ax.transAxes,
            va="top",
            color="#e5e7eb",
            fontsize=8.5,
            bbox={"facecolor": "#0b1220", "edgecolor": "#334155", "alpha": 0.9},
        )
        ax.set_xlim(0, 10)
        ax.set_ylim(500_000, ymax)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value / 1e6:.1f}M"))
        ax.set_xlabel("Years", color="#cbd5e1")
        ax.set_ylabel("Portfolio value", color="#cbd5e1")
        ax.set_title(
            "Weighted Portfolio Monte Carlo Forecast",
            color="#f8fafc",
            fontsize=13,
            weight="bold",
            pad=12,
        )
        ax.text(
            0.5,
            1.01,
            "500 correlated paths  |  25% funds  |  75% equities",
            transform=ax.transAxes,
            ha="center",
            color="#94a3b8",
            fontsize=8,
        )
        ax.grid(color="#334155", alpha=0.32, linewidth=0.7)
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.legend(loc="lower right", frameon=False, fontsize=8)

    movie = animation.FuncAnimation(
        fig, draw, frames=len(frame_indices), interval=75, repeat=True
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output_path, writer=animation.PillowWriter(fps=13))
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    make_animation(root / "assets" / "monte-carlo-portfolio.gif")
