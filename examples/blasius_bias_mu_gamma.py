import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc


rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 22})
rc("lines", linewidth=3)
rc('text.latex', preamble=r'\usepackage{amssymb}')

rougeCEA = "#b81420"
bleuEDF = "#10367a"
vertREML = "#008000"
orangeEDF = "#fe5716"


def ratio_to_true(est: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Ratio of each replication's estimate to the true value.

    est  : shape (n_rep, p)
    true : shape (p,)

    Returns
    -------
    ratio : shape (n_rep, p)
    """
    return est / true[None, :]


def plot_ratio_boxplots(param_name, symbol, true_diag, n_values, estimators, results_dir):
    p = true_diag.shape[0]

    # results[component][estimator] -> list of 1D arrays, one per n
    results = {comp: {est: [] for est in estimators} for comp in range(p)}

    for n in n_values:
        for est in estimators:
            data = pd.read_csv(
                os.path.join(results_dir, f"{param_name}_{est}_nrep_1000_n_{n}.csv")
            ).values
            data_est = data[1:, :]
            ratio = ratio_to_true(data_est, true_diag)
            for comp in range(p):
                results[comp][est].append(ratio[:, comp])

    colors = {"EM": bleuEDF, "ECME": rougeCEA, "REML": vertREML, "profile": orangeEDF}
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(estimators))

    for comp in range(p):
        fig, ax = plt.subplots(figsize=(12, 8))
        base_positions = np.arange(len(n_values))

        for est, offset in zip(estimators, offsets):
            data = results[comp][est]
            positions = base_positions + offset
            ax.boxplot(
                data,
                positions=positions,
                widths=width,
                patch_artist=True,
                boxprops=dict(facecolor=colors[est], alpha=0.7),
                medianprops=dict(color="black"),
                flierprops=dict(marker="o", markersize=3, alpha=0.5),
            )

        ax.set_xticks(base_positions)
        ax.set_xticklabels([f"n={n}" for n in n_values])
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel(rf"$\hat{{{symbol}}}_{comp + 1}/ {symbol}_{{{comp + 1}}}$")
        ax.set_xlabel(r"Sample size $n$")

        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[e], alpha=0.7) for e in estimators]
        ax.legend(handles, estimators, loc="best")

        #fig.tight_layout()
        fig.savefig(
            os.path.join(results_dir, f"boxplot_ratio_{param_name}_component_{comp}.pdf"),
            format="pdf",
            dpi=300,
        )
        plt.show()


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results_08092026")

    mu_true = np.array([np.log10(0.316) * 1.3, -0.25 * 1.3])
    gamma_true = np.diag(np.array([(0.1) ** 2, 0.7]))
    gamma_true_diag = np.diag(gamma_true)

    n_values = [5, 10, 15, 20, 25, 30]
    estimators = ["EM", "ECME", "REML", "profile"]

    plot_ratio_boxplots("mu", r"\mu", mu_true, n_values, estimators, results_dir)
    plot_ratio_boxplots("gamma", r"\gamma", gamma_true_diag, n_values, estimators, results_dir)


if __name__ == "__main__":
    main()
