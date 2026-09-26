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


def batched_mean_ratio(est: np.ndarray, true: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Split replications into consecutive batches of `batch_size`, average the
    estimated variance within each batch, and return the ratio of that
    batched mean to the true variance.

    est   : shape (n_rep, p)
    true  : shape (p,)

    Returns
    -------
    ratio : shape (n_rep // batch_size, p)
    """
    n_rep = est.shape[0]
    n_batches = n_rep // batch_size
    # Invalid (near-zero/negative) estimates are excluded from the batch average.
    est_valid = np.where(est > 1e-7, est, np.nan)
    batched = est_valid[: n_batches * batch_size].reshape(n_batches, batch_size, -1)
    batch_mean = np.nanmean(batched, axis=1)
    return batch_mean / true[None, :]

def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results_08092026")

    gamma_true = np.diag(np.array([(0.1) ** 2, 0.7]))
    gamma_true_diag = np.diag(gamma_true)
    p = gamma_true_diag.shape[0]

    n_values = [5, 10, 15, 20, 25, 30]  # fixed duplicate 20 -> 30; change if that was intentional
    estimators = ["EM", "ECME", "REML", "Profile"]
    batch_size = 10

    # results[component][estimator] -> list of 1D arrays, one per n
    results = {comp: {est: [] for est in estimators} for comp in range(p)}

    for n in n_values:
        gamma_em = pd.read_csv(os.path.join(results_dir, f"gamma_EM_nrep_1000_n_{n}.csv")).values
        gamma_ecme = pd.read_csv(os.path.join(results_dir, f"gamma_ECME_nrep_1000_n_{n}.csv")).values
        gamma_reml = pd.read_csv(os.path.join(results_dir, f"gamma_REML_nrep_1000_n_{n}.csv")).values
        gamma_profile = pd.read_csv(os.path.join(results_dir, f"gamma_profile_nrep_1000_n_{n}.csv")).values

        gamma_em_est = gamma_em[1:, :]
        gamma_ecme_est = gamma_ecme[1:, :]
        gamma_reml_est = gamma_reml[1:, :]
        gamma_profile_est = gamma_profile[1:, :]

        ratio_em = batched_mean_ratio(gamma_em_est, gamma_true_diag, batch_size)
        ratio_ecme = batched_mean_ratio(gamma_ecme_est, gamma_true_diag, batch_size)
        ratio_reml = batched_mean_ratio(gamma_reml_est, gamma_true_diag, batch_size)
        ratio_profile = batched_mean_ratio(gamma_profile_est, gamma_true_diag, batch_size)

        for comp in range(p):
            results[comp]["EM"].append(ratio_em[:, comp])
            results[comp]["ECME"].append(ratio_ecme[:, comp])
            results[comp]["REML"].append(ratio_reml[:, comp])
            results[comp]["Profile"].append(ratio_profile[:, comp])

    # --- Plotting: one figure per gamma component, grouped boxplots by n, colored by estimator ---
    colors = {"EM": bleuEDF, "ECME": rougeCEA, "REML": vertREML, "Profile": orangeEDF} 
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
        ax.set_ylabel(rf"$\mathbb{{E}}[\hat{{\gamma}}_{comp + 1}]/ \gamma_{{{comp + 1}}}$")
        ax.set_xlabel(r"Sample size $n$")
        #ax.set_title(f"$\gamma_{comp+1}$")

        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[e], alpha=0.7) for e in estimators]
        ax.legend(handles, estimators, loc="best")

        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"boxplot_ratio_component_{comp}.pdf"), format="pdf", dpi=300)
        plt.show()


if __name__ == "__main__":
    main()