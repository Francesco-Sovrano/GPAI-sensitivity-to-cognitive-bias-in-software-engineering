"""
Open-Ended SE Dilemmas — Bias Sensitivity Analysis

Loads two CSVs (one per model), analyzes Sensitivity_BASE vs Sensitivity_OURS,
and renders figures plus a summary table.

Defaults:
	--gpt_csv manually_analyzed_data/resolved_conflict/extracted_gpt4o_checked.csv
	--llama_csv manually_analyzed_data/resolved_conflict/extracted_llama_check.csv
	--out_dir  manually_analyzed_data

Usage example:
	python3 visualize_results.py \
		--gpt_csv manually_analyzed_data/resolved_conflict/extracted_gpt4o_checked.csv \
		--llama_csv manually_analyzed_data/resolved_conflict/extracted_llama_check.csv \
		--out_dir manually_analyzed_data
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smart_read_csv(path: Path) -> pd.DataFrame:
	"""Read CSV robustly by trying common separators and inference."""
	for sep in [",", ";", "\t", "|"]:
		try:
			return pd.read_csv(path, sep=sep, engine="python")
		except Exception:
			pass
	# Fallback to automatic separator inference
	return pd.read_csv(path, sep=None, engine="python")


def prep_dataframe(df: pd.DataFrame) -> pd.DataFrame:
		df = df.copy()
		for col in ["Sensitivity_BASE", "Sensitivity_OURS"]:
				df[col] = pd.to_numeric(df[col], errors="coerce")

		df = df.dropna(subset=["Sensitivity_BASE", "Sensitivity_OURS"])

		for col in ["Sensitivity_BASE", "Sensitivity_OURS"]:
				df[col] = df[col].clip(0, 1).round().astype(np.int8)

		print(len(df))
		return df


def bootstrap_ci_mean(arr: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 123):
	rng = np.random.default_rng(seed)
	n = len(arr)
	idx = np.arange(n)
	stats = []
	for _ in range(n_boot):
		res = rng.choice(idx, n, replace=True)
		stats.append(arr[res].mean())
	low = float(np.quantile(stats, alpha / 2))
	high = float(np.quantile(stats, 1 - alpha / 2))
	return (low, high)


def cohens_h(p1: float, p2: float) -> float:
	"""Effect size for proportions."""
	return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def mcnemar_counts(base: np.ndarray, ours: np.ndarray):
	"""Return discordant counts: improvements b (1->0) and regressions c (0->1)."""
	b = int(((base == 1) & (ours == 0)).sum())  # improvement
	c = int(((base == 0) & (ours == 1)).sum())  # regression
	return b, c


def mcnemar_pvalue(b: int, c: int, continuity: bool = True) -> float:
	"""Approximate p-value for McNemar's test (chi-square, 1 df)."""
	if b + c == 0:
		return float("nan")
	if continuity:
		chi2 = (abs(b - c) - 1) ** 2 / (b + c)
	else:
		chi2 = (b - c) ** 2 / (b + c)
	# P(Chi^2_1 >= x) = 2 * (1 - Phi(sqrt(x)))
	def normal_cdf(z):
		return 0.5 * (1 + math.erf(z / math.sqrt(2)))
	p = 2 * (1 - normal_cdf(math.sqrt(chi2)))
	return max(min(p, 1.0), 0.0)


def summarize_model(df: pd.DataFrame, model_name: str) -> dict:
	base = df["Sensitivity_BASE"].to_numpy()
	ours = df["Sensitivity_OURS"].to_numpy()
	n = len(df)

	mean_base = float(base.mean())
	mean_ours = float(ours.mean())
	abs_reduction = mean_base - mean_ours
	rel_reduction = (abs_reduction / mean_base) if mean_base > 0 else float("nan")

	# Discordant pairs
	b, c = mcnemar_counts(base, ours)
	ties_ok = int(((base == 0) & (ours == 0)).sum())
	ties_bad = int(((base == 1) & (ours == 1)).sum())

	# Bootstrap CIs for means and absolute reduction
	ci_base = bootstrap_ci_mean(base)
	ci_ours = bootstrap_ci_mean(ours)

	# For abs reduction CI, bootstrap paired difference (OURS - BASE)
	rng = np.random.default_rng(123)
	n_boot = 10000
	idx = np.arange(n)
	diffs = []
	for _ in range(n_boot):
		res = rng.choice(idx, n, replace=True)
		diffs.append(ours[res].mean() - base[res].mean())
	diffs = np.array(diffs)
	ci_abs_reduction = tuple(np.quantile(-diffs, [0.025, 0.975]))  # since diff=OURS-BASE

	# McNemar p-value (continuity corrected)
	mcnemar_p = mcnemar_pvalue(b, c, continuity=True)

	# Cohen's h
	h = cohens_h(mean_base, mean_ours)

	return {
		"model": model_name,
		"n_dilemmas": n,
		"mean_sensitivity_BASE": mean_base,
		"mean_sensitivity_OURS": mean_ours,
		"abs_reduction_pp": abs_reduction * 100,
		"rel_reduction_%": rel_reduction * 100 if not math.isnan(rel_reduction) else float("nan"),
		"BASE->OURS improvements (1->0)": b,
		"BASE->OURS regressions (0->1)": c,
		"both not sensitive (0->0)": ties_ok,
		"both sensitive (1->1)": ties_bad,
		"95% CI BASE": ci_base,
		"95% CI OURS": ci_ours,
		"95% CI abs reduction (pp)": (ci_abs_reduction[0] * 100, ci_abs_reduction[1] * 100),
		"McNemar p (continuity-corrected)": mcnemar_p,
		"Cohen's h (BASE vs OURS)": h,
	}



def plot_mean_bars(models: dict, out_path: Path):
	"""
	Compact grouped bar chart of mean bias sensitivity (BASE vs OURS) per model with bootstrap 95% CI.

	Changes vs the original:
	- grouped bars per model (shorter x-axis labels)
	- larger default fonts
	- narrower bars + tighter margins (more compact PDF)
	"""
	model_names = list(models.keys())

	base_means, ours_means = [], []
	base_err_low, base_err_high = [], []
	ours_err_low, ours_err_high = [], []

	for model_name in model_names:
		df = models[model_name]
		base = df["Sensitivity_BASE"].to_numpy()
		ours = df["Sensitivity_OURS"].to_numpy()

		m_base = float(base.mean())
		ci_base = bootstrap_ci_mean(base)
		base_means.append(m_base)
		base_err_low.append(m_base - ci_base[0])
		base_err_high.append(ci_base[1] - m_base)

		m_ours = float(ours.mean())
		ci_ours = bootstrap_ci_mean(ours)
		ours_means.append(m_ours)
		ours_err_low.append(m_ours - ci_ours[0])
		ours_err_high.append(ci_ours[1] - m_ours)

	x = np.arange(len(model_names))
	width = 0.32  # narrower bars -> less visual weight + more room for labels

	fig, ax = plt.subplots(figsize=(6.4, 3.4))

	# Typography
	ax.set_ylabel("Mean bias sensitivity (rate)", fontsize=15)
	ax.tick_params(axis="both", labelsize=13)

	bars_base = ax.bar(x - width / 2, base_means, width, label="Ø", color="steelblue")
	bars_ours = ax.bar(x + width / 2, ours_means, width, label="sAX+BW", color="darkorange")

	ax.errorbar(
		x - width / 2,
		base_means,
		yerr=[base_err_low, base_err_high],
		fmt="none",
		capsize=3,
		color="black",
		linewidth=1,
		zorder=3,
	)
	ax.errorbar(
		x + width / 2,
		ours_means,
		yerr=[ours_err_low, ours_err_high],
		fmt="none",
		capsize=3,
		color="black",
		linewidth=1,
		zorder=3,
	)

	ax.set_xticks(x)
	ax.set_xticklabels(model_names, fontsize=13)
	ax.set_ylim(0, 1)

	ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
	ax.set_axisbelow(True)
	ax.legend(frameon=False, ncol=2, fontsize=12, loc="upper right")

	# Value labels
	def _annotate(bars):
		for b in bars:
			h = float(b.get_height())
			y = min(h + 0.02, 0.98)  # keep text inside axes
			ax.text(
				b.get_x() + b.get_width() / 1.9,
				y,
				f"{h:.2f}",
				ha="center",
				va="bottom",
				fontsize=12,
			)

	_annotate(bars_base)
	_annotate(bars_ours)

	fig.tight_layout()
	fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
	plt.close(fig)
def plot_change_hist(df: pd.DataFrame, model_name: str, out_path: Path):
	"""Histogram (as a 3-bin bar chart) of per-dilemma change (OURS - BASE in {-1,0,+1})."""
	base = df["Sensitivity_BASE"].to_numpy()
	ours = df["Sensitivity_OURS"].to_numpy()
	diff = ours - base  # {-1, 0, +1}

	vals, counts = np.unique(diff, return_counts=True)
	bins = np.array([-1, 0, 1])
	y = [counts[vals.tolist().index(b)] if b in vals else 0 for b in bins]

	plt.figure(figsize=(8, 5))
	plt.bar(bins.astype(str), y)
	plt.xlabel("Per-Dilemma Change (OURS - BASE)")
	plt.ylabel("Count of Dilemmas")
	plt.title(f"{model_name}: Change in Bias Sensitivity per Dilemma\n(-1 improves, +1 regresses)")
	for xi, c in zip(bins.astype(str), y):
		plt.text(xi, c + 0.5, str(c), ha='center', va='bottom', fontsize=9)
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()


def plot_improve_regress(models: dict, out_path: Path):
	"""Grouped bars: improvements vs regressions per model."""
	improves = []
	regress = []
	model_labels = []

	for model_name, df in models.items():
		base = df["Sensitivity_BASE"].to_numpy()
		ours = df["Sensitivity_OURS"].to_numpy()
		b = int(((base == 1) & (ours == 0)).sum())  # improvement
		c = int(((base == 0) & (ours == 1)).sum())  # regression
		improves.append(b)
		regress.append(c)
		model_labels.append(model_name)

	x = np.arange(len(model_labels))
	width = 0.35

	plt.figure(figsize=(9, 6))
	plt.bar(x - width / 2, improves, width, label="Improvements (1→0)")
	plt.bar(x + width / 2, regress, width, label="Regressions (0→1)")
	plt.xticks(x, model_labels)
	plt.ylabel("Count of Dilemmas")
	plt.title("Discordant Pairs by Model: Improvements vs Regressions")
	plt.legend()
	for xi, a, b in zip(x, improves, regress):
		plt.text(xi - width / 2, a + 0.5, str(a), ha='center', va='bottom', fontsize=9)
		plt.text(xi + width / 2, b + 0.5, str(b), ha='center', va='bottom', fontsize=9)
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()


def format_summary_for_csv(summary_rows):
	"""Flatten CI tuples for CSV friendliness and return as DataFrame."""
	rows = []
	for row in summary_rows:
		r = dict(row)
		# Expand tuples into two columns (low/high)
		base_low, base_high = r.pop("95% CI BASE")
		ours_low, ours_high = r.pop("95% CI OURS")
		red_low, red_high = r.pop("95% CI abs reduction (pp)")

		r["95% CI BASE low"] = base_low
		r["95% CI BASE high"] = base_high
		r["95% CI OURS low"] = ours_low
		r["95% CI OURS high"] = ours_high
		r["95% CI abs reduction (pp) low"] = red_low
		r["95% CI abs reduction (pp) high"] = red_high
		rows.append(r)
	return pd.DataFrame(rows)


def main():
	parser = argparse.ArgumentParser(description="Bias Sensitivity Analysis (Open-Ended Dilemmas)")
	parser.add_argument("--gpt_csv", type=Path, default=Path("open_ended_dilemma_qualitative_analysis/manually_analyzed_data/resolved_conflict/extracted_gpt4o_checked.csv"))
	parser.add_argument("--llama_csv", type=Path, default=Path("open_ended_dilemma_qualitative_analysis/manually_analyzed_data/resolved_conflict/extracted_llama_checked.csv"))
	parser.add_argument("--out_dir", type=Path, default=Path("open_ended_dilemma_qualitative_analysis/data_visualization"))
	args = parser.parse_args()

	args.out_dir.mkdir(parents=True, exist_ok=True)

	# Load
	gpt_df = smart_read_csv(args.gpt_csv)
	llama_df = smart_read_csv(args.llama_csv)

	# Prep
	gpt = prep_dataframe(gpt_df)
	llama = prep_dataframe(llama_df)

	models = {
		"gpt-4o-mini": gpt,
		"llama-3.1-8b": llama,
	}

	# Summaries
	summaries = [summarize_model(gpt, "gpt-4o-mini"),
				 summarize_model(llama, "llama-3.1-8b")]
	summary_df = format_summary_for_csv(summaries)

	# Save summary table
	summary_csv = args.out_dir / "bias_sensitivity_summary.csv"
	summary_df.to_csv(summary_csv, index=False)

	# Print concise console summary
	print("\n=== Bias Sensitivity Summary (Open-Ended Dilemmas) ===")
	for s in summaries:
		print(f"\nModel: {s['model']}  (n={s['n_dilemmas']})")
		print(f"  Mean sensitivity BASE → OURS: {s['mean_sensitivity_BASE']:.3f} → {s['mean_sensitivity_OURS']:.3f}")
		print(f"  Absolute reduction (pp): {s['abs_reduction_pp']:.1f}  "
				f"[95% CI: {s['95% CI abs reduction (pp)'][0]:.1f}, {s['95% CI abs reduction (pp)'][1]:.1f}]")
		rr = s['rel_reduction_%']
		print(f"  Relative reduction: {rr:.1f}%") if not math.isnan(rr) else print("  Relative reduction: NA")
		print(f"  Improvements (1→0): {s['BASE->OURS improvements (1->0)']}  |  Regressions (0→1): {s['BASE->OURS regressions (0->1)']}")
		print(f"  McNemar p (continuity-corrected): {s['McNemar p (continuity-corrected)']:.4f}")
		print(f"  Cohen's h:", s["Cohen's h (BASE vs OURS)"])

	# Figures
	plot_mean_bars(models, args.out_dir / "mean_sensitivity_by_strategy.pdf")
	plot_change_hist(gpt, "gpt-4o-mini", args.out_dir / "change_hist_gpt.pdf")
	plot_change_hist(llama, "llama-3.1-8b", args.out_dir / "change_hist_llama.pdf")
	plot_improve_regress(models, args.out_dir / "improvements_vs_regressions.pdf")

	print("\nSaved:")
	print(f"  {summary_csv}")
	print(f"  {args.out_dir / 'mean_sensitivity_by_strategy.pdf'}")
	print(f"  {args.out_dir / 'change_hist_gpt.pdf'}")
	print(f"  {args.out_dir / 'change_hist_llama.pdf'}")
	print(f"  {args.out_dir / 'improvements_vs_regressions.pdf'}")


if __name__ == "__main__":
	main()
