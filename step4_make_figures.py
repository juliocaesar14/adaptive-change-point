# Step 4, draw the charts and write the tables used in the report.

import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config
from src.methods.method_1_fixed_smoothing import fixed_smooth
from src.methods.method_2_adaptive_smoothing import adaptive_smooth
from src.methods.method_3_adaptive_window import adaptive_window_line
from step1_prepare_data import load_log_prices

BLUE, ORANGE, GREY, DARK = "#2a78d6", "#c1602c", "#8a8984", "#2b2a28"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": DARK, "xtick.color": DARK,
                     "ytick.color": DARK, "figure.dpi": 100, "savefig.dpi": 220})


def save(fig, name):
    # Save one figure into the results folder.
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, name))
    plt.close(fig)


def price_figure(dates, y, split):
    # S and P 500 with the hindsight answer key.
    points = pd.read_csv(os.path.join(config.RESULTS_DIR, "sp500_turning_points.csv"), parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(7, 2.9))
    ax.plot(dates, y, color=BLUE, linewidth=1.1, label="Log price")
    ax.scatter(points["date"], points["log_price"], s=16, color=ORANGE, zorder=3, label="Turning point")
    ax.axvline(dates[split], color=GREY, linestyle="--", linewidth=1)
    ax.text(dates[split], y.max(), " testing starts", va="top", color=DARK)
    ax.set_ylabel("Log of S and P 500")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "sp500_turning_points.png")


def smoothing_figure(dates, y, settings):
    # The COVID crash, fixed and adaptive lines, then alpha over time.
    low, high = np.searchsorted(dates, pd.Timestamp("2019-10-01")), np.searchsorted(dates, pd.Timestamp("2020-08-01"))
    fixed = fixed_smooth(y, 0.05)
    a = settings["Adaptive smoothing"]
    adaptive, alpha = adaptive_smooth(y, a["alpha_min"], a["alpha_max"])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 3.8), sharex=True, height_ratios=[3, 1])
    top.plot(dates[low:high], y[low:high], color=GREY, linewidth=1, label="Actual")
    top.plot(dates[low:high], fixed[low:high], color=ORANGE, linewidth=1.8, label="Fixed, alpha 0.05")
    top.plot(dates[low:high], adaptive[low:high], color=BLUE, linewidth=1.8, label="Adaptive")
    top.set_ylabel("Log price")
    top.legend(frameon=False, ncol=3, loc="lower left")
    bottom.plot(dates[low:high], alpha[low:high], color=BLUE, linewidth=1.5)
    bottom.set_ylabel("Adaptive alpha")
    save(fig, "smoothing_comparison.png")


def window_figure(dates, y, settings):
    # How the window length reacts through time.
    a = settings["Adaptive window"]
    _, lengths = adaptive_window_line(y, a["min_length"], a["max_length"], a["surprise"])
    low = np.searchsorted(dates, pd.Timestamp("2019-10-01"))
    high = np.searchsorted(dates, pd.Timestamp("2020-08-01"))
    fig, ax = plt.subplots(figsize=(7, 1.9))
    ax.step(dates[low:high], lengths[low:high], color=BLUE, linewidth=1.3)
    ax.set_ylabel("Window length, days")
    save(fig, "window_length.png")


def simulation_figure():
    # Average F1 with 95 percent intervals, one panel per fake market.
    table = pd.read_csv(os.path.join(config.RESULTS_DIR, "simulation_scores.csv"))
    cases = list(dict.fromkeys(table["case"]))
    methods = list(dict.fromkeys(table["method"]))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True, sharex=True)
    for ax, case in zip(axes, cases):
        part = table[table["case"] == case].set_index("method").loc[methods]
        ax.errorbar(part["f1"], range(len(methods)), xerr=[part["f1"] - part["f1_low"], part["f1_high"] - part["f1"]],
                    fmt="o", color=BLUE, markersize=5, capsize=2)
        ax.set_title(case, fontsize=9)
        ax.set_xlabel("Average F1")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", color="#e6e5e0")
    axes[0].set_yticks(range(len(methods)), methods)
    save(fig, "simulation_f1.png")


def alarm_figure(dates, split):
    # One row of alarm marks for each method on the test years.
    with open(os.path.join(config.RESULTS_DIR, "sp500_alarms.json")) as handle:
        alarms = json.load(handle)
    points = pd.read_csv(os.path.join(config.RESULTS_DIR, "sp500_turning_points.csv"), parse_dates=["date"])
    points = points[points["date"] >= dates[split]]
    names = list(alarms)[::-1]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for date in points["date"]:
        ax.axvline(date, color=ORANGE, linewidth=0.9, alpha=0.7)
    for row, name in enumerate(names):
        days = pd.to_datetime([a["date"] for a in alarms[name]])
        days = days[days >= dates[split]]
        ax.scatter(days, [row] * len(days), marker="|", s=70, color=BLUE, linewidths=1.5)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(dates[split], dates[-1])
    ax.set_title("Blue marks are alarms, orange lines are hindsight turning points", fontsize=9, loc="left")
    save(fig, "sp500_alarms.png")


def tex_tables():
    # Write report tables directly from the saved results.
    out = os.path.join(config.ROOT, "report", "tables")
    os.makedirs(out, exist_ok=True)
    sim = pd.read_csv(os.path.join(config.RESULTS_DIR, "simulation_scores.csv"))
    lines = []
    for case in dict.fromkeys(sim["case"]):
        lines.append(f"\\midrule\n\\multicolumn{{6}}{{l}}{{\\textit{{Case {case}}}}}\\\\")
        for _, r in sim[sim["case"] == case].iterrows():
            lines.append(f"{r['method']} & {r['f1']:.3f} [{r['f1_low']:.3f}, {r['f1_high']:.3f}] & {r['precision']:.3f} & "
                         f"{r['recall']:.3f} & {r['mean_delay']:.1f} & {r['false_per_1000']:.1f}\\\\")
    open(os.path.join(out, "simulation.tex"), "w").write("\n".join(lines))
    tests = pd.read_csv(os.path.join(config.RESULTS_DIR, "simulation_paired_tests.csv"))
    lines = []
    for _, r in tests.iterrows():
        p = "$<0.001$" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}"
        lines.append(f"{r['case'].split()[0]} & {r['first']} vs {r['second']} & {r['mean_difference']:+.3f} & {p}\\\\")
    open(os.path.join(out, "paired.tex"), "w").write("\n".join(lines))
    real = pd.read_csv(os.path.join(config.RESULTS_DIR, "sp500_scores.csv"))
    for period in ("tune", "test"):
        lines = []
        for _, r in real[real["period"] == period].iterrows():
            p = "$<0.001$" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}"
            lines.append(f"{r['method']} & {int(r['alarms'])} & {int(r['hits'])} & {r['precision']:.3f} & {r['recall']:.3f} & "
                         f"{r['f1']:.3f} & {r['mean_delay']:.1f} & {r['random_f1']:.3f} & {p}\\\\")
        open(os.path.join(out, f"sp500_{period}.tex"), "w").write("\n".join(lines))
    sens = pd.read_csv(os.path.join(config.RESULTS_DIR, "sp500_sensitivity.csv"))
    lines = []
    for name in dict.fromkeys(sens["method"]):
        part = sens[sens["method"] == name]
        cells = [f"{part[(part.turn_size == s) & (part.tolerance == t)].f1.iloc[0]:.2f}" for s in (0.07, 0.10, 0.15) for t in (20, 40, 60)]
        lines.append(name + " & " + " & ".join(cells) + "\\\\")
    open(os.path.join(out, "sensitivity.tex"), "w").write("\n".join(lines))
    # The machine learning scores come from the Colab notebook, so they may be absent.
    ml_path = os.path.join(config.RESULTS_DIR, "ml_scores.csv")
    if not os.path.exists(ml_path):
        return
    ml = pd.read_csv(ml_path)
    lines = []
    for _, r in ml.iterrows():
        ap = "--" if pd.isna(r["average_precision"]) else f"{r['average_precision']:.3f}"
        p = "$<0.001$" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}"
        lines.append(f"{r['model']} & {int(r['alarms'])} & {int(r['hits'])} & {r['precision']:.3f} & {r['recall']:.3f} & "
                     f"{r['f1']:.3f} & {ap} & {p}\\\\")
    open(os.path.join(out, "ml.tex"), "w").write("\n".join(lines))


def main(source):
    # Draw everything.
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    dates, y, _ = load_log_prices(source)
    split = int(np.searchsorted(dates, pd.Timestamp(config.SOURCES[source]["split"])))
    with open(os.path.join(config.RESULTS_DIR, "sp500_settings.json")) as handle:
        settings = json.load(handle)
    price_figure(dates, y, split)
    smoothing_figure(dates, y, settings)
    window_figure(dates, y, settings)
    simulation_figure()
    alarm_figure(dates, split)
    tex_tables()
    print("Figures and tables saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="bundled", choices=["bundled", "yahoo"])
    main(parser.parse_args().source)
