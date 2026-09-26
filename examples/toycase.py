import numpy as np
import pandas as pd
import pycirce as pyc
import CoolProp as cp
import scipy.stats as st
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from matplotlib import rc

rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 22})
rc("lines", linewidth=3)
rougeCEA = "#b81420"
orangeEDF = "#fe5716"
bleuEDF = "#10367a"


def toy_case(x):
    """
    Helper function that computes the toy case
    """

    return np.log10(0.316) - 0.25 * x


def generate_data_toycase(mu, gamma, data, seed):
    """
    Generate toy case data according to the design of experiments 
    """
    np.random.seed(seed)

    n = data.shape[0]

    theta = np.random.multivariate_normal(mu, gamma, size=n)

    return theta[:, 0] + theta[:, 1] * data

n_rep = 2

mu_em = np.zeros((n_rep + 1, 2))
gamma_em = np.zeros((n_rep + 1, 2))

mu_ecme = np.zeros((n_rep + 1, 2))
gamma_ecme = np.zeros((n_rep + 1, 2))

mu_reml = np.zeros((n_rep + 1, 2))
gamma_reml = np.zeros((n_rep + 1, 2))

mu_profile = np.zeros((n_rep + 1, 2))
gamma_profile = np.zeros((n_rep + 1, 2))

gamma_true = np.diag(np.array([0.5, 0.7]))
mu_true = np.array([np.log10(0.316) * 1.3, -0.25 * 1.3])

mu_em[0, :] = mu_true
gamma_em[0, :] = np.diag(gamma_true)

mu_ecme[0, :] = mu_true
gamma_ecme[0, :] = np.diag(gamma_true)

mu_reml[0, :] = mu_true
gamma_reml[0, :] = np.diag(gamma_true)

mu_profile[0, :] = mu_true
gamma_profile[0, :] = np.diag(gamma_true)


log_alpha_range = np.linspace(-3, 0, num=50)

bic_array = np.zeros(len(log_alpha_range))

n = 5

T = np.array([25, 110, 250])
P = np.array([60, 90, 110, 130])

data = np.random.normal(size=n)

def run_replication(it):
    """
    Run one Monte-Carlo replication: generate noisy data for replication `it`,
    fit EM/ECME/REML/profile estimators, and compute their coverage/interquantile
    diagnostics. Reads module-level globals (n, data, mu_true, gamma_true, cov_rate)
    that are reconstructed identically in each worker process.
    """
    print(f"it = {it}")

    np.random.seed(it)

    sig_eps = np.random.normal(0, 1, n)

    f_toycase_exp = generate_data_toycase(mu_true, gamma_true, data, it) + sig_eps

    h = np.zeros((2, n))
    h[0, :] = np.repeat(1.0, n)
    h[1, :] = data 

    bmin = 1/3.0 * np.amax((1) ** 2/(h[0, :] ** 2 + h[1, :] ** 2))


    circe_diag = pyc.CirceEMdiag(
        initial_mean=[np.log10(0.316), -0.25],
        initial_cov= bmin * np.identity(2),
        h=h,
        z_exp=f_toycase_exp ,
        z_nom=np.repeat(0, n),
        sig_eps=sig_eps,
        niter=15000,
        tolerance=1e-5,
    )

    circe_diag_ecme = pyc.CirceECMEdiag(
        initial_mean=[np.log10(0.316), -0.25],
        initial_cov= bmin * np.identity(2),
        h=h,
        z_exp=f_toycase_exp ,
        z_nom=np.repeat(0, n),
        sig_eps=sig_eps,
        niter=15000,
        tolerance=1e-5,
    )

    circe_reml = pyc.CirceREML(
        initial_mean=[np.log10(0.316), -0.25],
        initial_cov= bmin * np.identity(2),
        h=h,
        z_exp=f_toycase_exp ,
        z_nom=np.repeat(0, n),
        sig_eps=sig_eps,
        niter=15000
    )


    circe_profile = pyc.CirceProfile(
        initial_mean=[np.log10(0.316), -0.25],
        initial_cov= bmin * np.identity(2),
        h=h,
        z_exp=f_toycase_exp ,
        z_nom=np.repeat(0, n),
        sig_eps=sig_eps,
        niter=15000
    )


    mu1, gamma1, _, _ = circe_diag_ecme.estimate(n_starts=10)

    mu2, gamma2, _, _ = circe_diag.estimate(n_starts=10)

    mu3, gamma3, _ = circe_reml.estimate(n_starts=10)
    mu4, gamma4, _ = circe_profile.estimate(n_starts=10)


    mu_em_it = np.array(mu2[-1])[:, 0]
    gamma_em_it = np.diag(gamma2[-1])

    mu_ecme_it = np.array(mu1[-1])[:, 0]
    gamma_ecme_it = np.diag(gamma1[-1])

    mu_reml_it = mu3
    gamma_reml_it = gamma3

    mu_profile_it = mu4
    gamma_profile_it = gamma4

    return {
        "mu_em": mu_em_it,
        "gamma_em": gamma_em_it,
        "mu_ecme": mu_ecme_it,
        "gamma_ecme": gamma_ecme_it,
        "mu_reml": mu_reml_it,
        "gamma_reml": gamma_reml_it,
        "mu_profile": mu_profile_it,
        "gamma_profile": gamma_profile_it,
    }


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_replication, range(1, n_rep + 1)))

    for i, res in enumerate(results):
        it = i + 1

        mu_em[it, :] = res["mu_em"]
        gamma_em[it, :] = res["gamma_em"]

        mu_ecme[it, :] = res["mu_ecme"]
        gamma_ecme[it, :] = res["gamma_ecme"]

        mu_reml[it, :] = res["mu_reml"]
        gamma_reml[it, :] = res["gamma_reml"]

        mu_profile[it, :] = res["mu_profile"]
        gamma_profile[it, :] = res["gamma_profile"]


    pd.DataFrame(mu_em).to_csv(f"./results_toycase/mu__EM_nrep_{n_rep}_n_{n}.csv", index=False)
    pd.DataFrame(gamma_em).to_csv(f"./results_toycase/gamma_EM_nrep_{n_rep}_n_{n}.csv", index=False)

    pd.DataFrame(mu_ecme).to_csv(f"./results_toycase/mu_ECME_nrep_{n_rep}_n_{n}.csv", index=False)
    pd.DataFrame(gamma_ecme).to_csv(f"./results_toycase/gamma_ECME_nrep_{n_rep}_n_{n}.csv", index=False)

    pd.DataFrame(mu_reml).to_csv(f"./results_toycase/mu_REML_nrep_{n_rep}_n_{n}.csv", index=False)
    pd.DataFrame(gamma_reml).to_csv(f"./results_toycase/gamma_REML_nrep_{n_rep}_n_{n}.csv", index=False)


    pd.DataFrame(mu_profile).to_csv(f"./results_toycase/mu_profile_nrep_{n_rep}_n_{n}.csv", index=False)
    pd.DataFrame(gamma_profile).to_csv(f"./results_toycase/gamma_profile_nrep_{n_rep}_n_{n}.csv", index=False)



