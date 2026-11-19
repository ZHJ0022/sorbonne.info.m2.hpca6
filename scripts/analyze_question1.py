"""Plot Question 1 results and errors."""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R0, A, SIGMA = 0.012, 1.0, 0.1  # Model parameters.


def reference(t):
    # m = E[r(t)], J = E[integral_0^t r(s) ds].
    m, J = R0, 0.0
    # Integrate the two linear segments of theta.
    for start, end, theta_start, slope in (
        (0.0, min(t, 5.0), 0.012, 0.0014),
        (5.0, t, 0.019, 0.001),
    ):
        h = end - start
        if h <= 0:
            continue
        b = -math.expm1(-A * h) / A
        c = (h - b) / A
        J += m * b + theta_start * c + slope * (h * h / (2 * A) - c / A)
        m = m * math.exp(-A * h) + theta_start * b + slope * c

    # Integral variance.
    u = -math.expm1(-A * t)
    variance = SIGMA**2 / A**2 * (
        t - 2 * u / A - math.expm1(-2 * A * t) / (2 * A)
    )
    # Gaussian formulas: P=exp(-J+Var(I)/2), f=m-Cov(r,I).
    return (
        math.exp(-J + variance / 2),
        m - SIGMA**2 * u**2 / (2 * A**2),
    )


def main():
    # Read the existing CSV.
    time, price, forward = [], [], []
    with (ROOT / "question1.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            time.append(float(row["T"]))
            price.append(float(row["P"]))
            forward.append(float(row["f"]))

    # Compute reference curves.
    exact_price, exact_forward = [], []
    for t in time:
        p, f = reference(t)
        exact_price.append(p)
        exact_forward.append(f)

    summary = (
        "Question 1: existing CSV vs analytical reference\n"
        "Model: r0=0.012, a=1, sigma=0.1; theta(t) from the subject.\n"
        f"Grid: {len(time)} points, dt={time[1] - time[0]:g}.\n\n"
        "Curve: maximum absolute error, RMSE\n"
    )
    # Plot curves and errors.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, (name, values, exact) in enumerate((
        ("P(0,T)", price, exact_price),
        ("f(0,T)", forward, exact_forward),
    )):
        # Compute error statistics.
        errors = [values[i] - exact[i] for i in range(len(time))]
        maximum = max(abs(x) for x in errors)
        rmse = math.sqrt(sum(x * x for x in errors) / len(errors))
        summary += f"{name}: {maximum:.9g}, {rmse:.9g}\n"
        axes[0, col].plot(time, values, label="Existing Monte Carlo CSV")
        axes[0, col].plot(time, exact, "--", label="Analytical reference")
        axes[0, col].set_title(name)
        axes[0, col].legend()
        axes[1, col].plot(time, errors)
        axes[1, col].axhline(0, color="black", linewidth=0.7)
        axes[1, col].set_title(name + ": CSV - analytical")
        axes[1, col].set_xlabel("Maturity T (years)")
        axes[1, col].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()

    # Save the figure and statistics.
    summary += "\nErrors include sampling and trapezoidal integration errors.\n"
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "question1.png", dpi=160)
    plt.close(fig)
    (output / "question1_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
