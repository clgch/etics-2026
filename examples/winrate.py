import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

# LaTeX styling for plots
rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 22})
rc("lines", linewidth=3)

rougeCEA = "#b81420"
bleuEDF = "#10367a"
vertECME = "#2ca02c"
orangeREML = "#ff7f0e"

def relative_mae_score(est: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Compute the log10 of the mean absolute relative error.
    Used as the 'score' to determine the winner.
    """
    return np.abs(est[:, 1] - true[1])
def main():
    # Project root is the parent directory of this 'examples' folder
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")

    # True parameters
    # True parameters (must match those in blasius.py)
    gamma_true = np.diag(np.array([(0.1)**2, 0.7]))
    mu_true = np.array([np.log10(0.316) * 1.3, -0.25 * 1.3])
    gamma_true_diag = np.diag(gamma_true)

    n_seeds = 500
    methods = ["EM", "Profile", "ECME", "REML"]
    
    mu_results = {m: np.zeros((n_seeds, 2)) for m in methods}
    gamma_results = {m: np.zeros((n_seeds, 2)) for m in methods}

    # Load data
    print("Loading data...")
    # Load results (filenames are those written in blasius.py)
    mu_results["EM"] = pd.read_csv(os.path.join(results_dir, "mu_em_nrep_500.csv")).values[1:, :]
    gamma_results["EM"] = pd.read_csv(os.path.join(results_dir, "gamma_em_nrep_500.csv")).values[1:, :]
    mu_results["ECME"] = pd.read_csv(os.path.join(results_dir, "mu_ecme_nrep_500.csv")).values[1:, :]
    gamma_results["ECME"] = pd.read_csv(
        os.path.join(results_dir, "gamma_ecme_nrep_500.csv")
    ).values[1:, :]
    mu_results["REML"] = pd.read_csv(os.path.join(results_dir, "mu_reml_nrep_500.csv")).values[1:, :]
    gamma_results["REML"] = pd.read_csv(os.path.join(results_dir, "gamma_reml_nrep_500.csv")).values[1:, :]
    mu_results["Profile"] = pd.read_csv(os.path.join(results_dir, "mu_profile_nrep_500.csv")).values[1:, :]
    gamma_results["Profile"] = pd.read_csv(os.path.join(results_dir, "gamma_profile_nrep_500.csv")).values[1:, :]


    # Compute scores (lower is better)
    scores_mu = np.zeros((n_seeds, 4))
    scores_gamma = np.zeros((n_seeds, 4))

    for idx, method in enumerate(methods):
        scores_mu[:, idx] = relative_mae_score(mu_results[method], mu_true)
        scores_gamma[:, idx] = relative_mae_score(gamma_results[method], gamma_true_diag)

    # Combined score (mean of the two log-scores)
    scores_combined = (scores_mu + scores_gamma) / 2.0

    def get_winrates(scores):
        # Find index of minimum score for each row (seed)
        winners = np.argmin(scores, axis=1)
        # Count occurrences of each winner index
        counts = np.bincount(winners, minlength=4)
        return (counts / n_seeds) * 100

    winrate_mu = get_winrates(scores_mu)
    winrate_gamma = get_winrates(scores_gamma)
    winrate_combined = get_winrates(scores_combined)

    print("\nWin Rates (%):")
    for idx, method in enumerate(methods):
        print(f"{method:5}: Mu={winrate_mu[idx]:.1f}%, Gamma={winrate_gamma[idx]:.1f}%, Combined={winrate_combined[idx]:.1f}%")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    x = np.arange(len(methods))
    bar_colors = [bleuEDF, rougeCEA, vertECME, orangeREML]
    
    # MU Win Rate
    ax1.bar(methods, winrate_mu, color=bar_colors, alpha=1, edgecolor='black')
    ax1.set_ylabel(r"Win Rate $(\%)$")
    ax1.set_title(r"$m$")
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # Add text labels for Mu
    for i, v in enumerate(winrate_mu):
        ax1.text(i, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')

    # GAMMA Win Rate
    ax2.bar(methods, winrate_gamma, color=bar_colors, alpha=1, edgecolor='black')
    ax2.set_ylabel(r"Win Rate $(\%)$")
    ax2.set_title(r"$\Gamma$")
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)

    # Add text labels for Gamma
    for i, v in enumerate(winrate_gamma):
        ax2.text(i, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')

    # Legend (just to clarify colors if needed, though labels are on X axis)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=bleuEDF, alpha=1, label="EM"),
        Patch(facecolor=rougeCEA, alpha=1, label="Profile"),
        Patch(facecolor=vertECME, alpha=1, label="ECME"),
        Patch(facecolor=orangeREML, alpha=1, label="REML"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for legend
    plt.savefig("winrates_mu_gamma_sep.pdf", dpi=300, format="pdf")
    plt.show()

if __name__ == "__main__":
    main()
