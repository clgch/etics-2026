#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagonal-covariance ECME algorithm for probabilistic inversion of
thermohydraulic correlations (pure Python, vectorised NumPy).
"""

import numpy as np


class CirceECMEdiag:
    """
    Expectation Conditional Maximisation Either (ECME) algorithm for
    probabilistic inversion of thermohydraulic correlation, with
    diagonal covariance matrix.
    """

    def __init__(
        self,
        initial_mean=None,
        initial_cov=None,
        tolerance=None,
        h=None,
        z_exp=None,
        z_nom=None,
        sig_eps=None,
        niter=None,
    ):
        # Missing parameters
        if z_exp is None:
            raise ValueError("Please provide a vector of experimental values in z_exp")
        if h is None:
            raise ValueError(
                "Please provide a Jacobian matrix h of size "
                "(n_parameters, n_observations) of the thermohydraulic numerical simulator"
            )
        if z_nom is None:
            raise ValueError("Please provide a vector of nominal simulations in z_nom")
        if sig_eps is None:
            raise ValueError(
                "Please provide a vector of measurement standard deviations in sig_eps"
            )

        # Cast to NumPy arrays
        h = np.asarray(h, dtype=float)
        z_exp = np.asarray(z_exp, dtype=float)
        z_nom = np.asarray(z_nom, dtype=float)
        sig_eps = np.asarray(sig_eps, dtype=float)

        # Check dataset size consistency
        if not (h.shape[1] == len(z_exp) == len(z_nom) == len(sig_eps)):
            raise ValueError(
                "ERROR: size inconsistencies between the Jacobian h, "
                "the vectors z_exp, z_nom and sig_eps!"
            )

        p = h.shape[0]

        # Initial mean / covariance (full diagonal matrix for public API)
        if initial_mean is None:
            self.initial_mean = np.ones(p).reshape(-1, 1)
        else:
            self.initial_mean = np.array(initial_mean, dtype=float).reshape(-1, 1)

        if initial_cov is None:
            self.initial_cov = np.diag(np.ones(p, dtype=float))
        else:
            self.initial_cov = np.array(initial_cov, dtype=float)

        # Tolerance and iteration limit
        self.tolerance = 1e-5 if tolerance is None else float(tolerance)

        self.h = h
        self.z_exp = z_exp
        self.z_nom = z_nom
        self.sig_eps = sig_eps
        self.niter = niter

        # Internal diagonal representation
        self._mean0 = self.initial_mean.reshape(-1).astype(float)
        self._gamma0 = np.diag(self.initial_cov).astype(float)

    @staticmethod
    def _one_step_ecme_diag(mean, gamma, h, z_exp, z_nom, sig_eps):
        """
        One ECME step using only the diagonal of the covariance (gamma).

        Parameters
        ----------
        mean  : (p,)
        gamma : (p,)
        h     : (p, n)
        """
        p, n = h.shape

        # E-step: compute b_i and S_i as in EM, but maintain gamma as diagonal
        resid = z_exp - z_nom - np.sum(h * mean[:, None], axis=0)  # (n,)
        denom = sig_eps**2 + np.sum(gamma[:, None] * h**2, axis=0)  # (n,)

        g = gamma[:, None] * h  # (p, n)
        b = g * resid[None, :] / denom[None, :]  # (p, n)
        S_diag = (g**2) / denom[None, :]  # (p, n)
        delta_cov_diag_mat = b**2 - S_diag  # (p, n)

        delta_cov_diag = np.mean(delta_cov_diag_mat, axis=1)  # (p,)
        gamma_new = gamma + delta_cov_diag  # ECME: no -delta_mean^2 term

        # M-step for mean (exact maximisation):
        # Build H_tilde and its pseudoinverse with diagonal gamma_new
        # H_tilde = sum_i (h_i h_i^T / (h_i^T cov_new h_i + sig_eps_i^2))
        # with cov_new diagonal(gamma_new)
        w = 1.0 / (sig_eps**2 + np.sum(gamma_new[:, None] * h**2, axis=0))  # (n,)

        # Weighted outer products: sum_i w_i * h_i h_i^T
        # h has shape (p, n); apply weights along the observation axis
        H_tilde = (h * w[None, :]) @ h.T  # (p, p)

        # Moore–Penrose inverse of H_tilde via SVD
        U, s, Vt = np.linalg.svd(H_tilde, full_matrices=False)
        s_pseudo = np.zeros_like(s)
        non_zero = s > 1e-10
        s_pseudo[non_zero] = 1.0 / s[non_zero]
        H_plus = Vt.T @ np.diag(s_pseudo) @ U.T

        # Right-hand side: sum_i h_i * (z_exp[i] - z_nom[i]) / denom_i
        rhs = h @ ((z_exp - z_nom) * w)  # (p,)

        mean_new = H_plus @ rhs  # (p,)

        return mean_new, gamma_new

    @staticmethod
    def _loglik_diag(mean, gamma, h, z_exp, z_nom, sig_eps):
        """
        Observed-data log-likelihood with diagonal covariance.
        """
        resid = z_exp - z_nom - np.sum(h * mean[:, None], axis=0)  # (n,)
        denom = sig_eps**2 + np.sum(gamma[:, None] * h**2, axis=0)  # (n,)
        loglik = -0.5 * np.sum(resid**2 / denom) + 0.5 * np.sum(np.log(denom))
        return loglik

    def _run_single_ecme(self, init_mu, init_gamma):
            """
            Run one ECME trajectory to convergence from a given starting point.
    
            Returns
            -------
            mean_list, cov_list, loglik_list : list
                Trajectories, all of equal length (one entry per iterate,
                including the initial point).
            iterator : int
                Number of EM steps taken after the initial point.
            """
            eps = 1e-12
    
            cov_list = [np.diag(init_gamma)]
            mean_list = [init_mu.reshape(-1, 1)]
            loglik_list = [
                self._loglik_diag(
                    init_mu, init_gamma, self.h, self.z_exp, self.z_nom, self.sig_eps
                )
            ]
    
            mean, gamma = init_mu, init_gamma
            iterator = 0
            err_cov = np.inf
            err_mean = np.inf
    
            while (err_cov > self.tolerance or err_mean > self.tolerance) and (
                self.niter is None or iterator < self.niter
            ):
                mean_new, gamma_new = self._one_step_ecme_diag(
                    mean, gamma, self.h, self.z_exp, self.z_nom, self.sig_eps
                )
    
                cov_list.append(np.diag(gamma_new))
                mean_list.append(mean_new.reshape(-1, 1))
                loglik_list.append(
                    self._loglik_diag(
                        mean_new, gamma_new, self.h, self.z_exp, self.z_nom, self.sig_eps
                    )
                )
    
                rel_diff_cov = np.abs(gamma_new - gamma) / np.maximum(np.abs(gamma_new), eps)
                err_cov = float(np.max(rel_diff_cov))
    
                rel_diff_mean = np.abs(mean_new - mean) / np.maximum(np.abs(mean_new), eps)
                err_mean = float(np.max(rel_diff_mean))
    
                mean, gamma = mean_new, gamma_new
                iterator += 1
    
            return mean_list, cov_list, loglik_list, iterator
    
    def estimate(self, n_starts=10, random_state=None):
        """
        Run the ECME iterations until convergence (or until niter is reached).

        Returns
        -------
        mean_list : list of ndarray
            Sequence of mean vectors.
        cov_list : list of ndarray
            Sequence of (diagonal) covariance matrices.
        loglik_list : list of float
            Log-likelihood values at each iteration.
        err_cov_list : list of float
            Max relative change of the covariance diagonal at each iteration.
        err_mean_list : list of float
            Max relative change of the mean vector at each iteration.
        """
        p = self.h.shape[0]
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

        best_fun = -np.inf
        best_res = None

        for i in range(n_starts):
            if i == 0:
                init_mu = self._mean0.copy()
                init_gamma = self._gamma0.copy()
            else:
                scale = np.mean(self._gamma0) if np.sum(self._gamma0) > 1e-9 else 1.0
                init_gamma = rng.uniform(0.1 * scale, 10 * scale, size=p)
                init_mu = rng.multivariate_normal(self._mean0, np.diag(init_gamma))

            mean_list, cov_list, loglik_list, iterator = self._run_single_ecme(
                init_mu, init_gamma
            )

            final_loglik = loglik_list[-1]
            if not np.isfinite(final_loglik):
                continue

            if final_loglik > best_fun:
                best_fun = final_loglik
                best_res = (mean_list, cov_list, loglik_list, iterator)
        if best_res is None:
            raise RuntimeError(
                "ECME failed to converge to a finite log-likelihood from any of the "
                f"{n_starts} starting points."
            )

        mean_list, cov_list, loglik_list, iterator = best_res

        return mean_list, cov_list, loglik_list, iterator
