#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagonal-covariance EM algorithm for probabilistic inversion of
thermohydraulic correlations (pure Python, vectorised NumPy).
"""

import numpy as np


class CirceEMdiag:
    """
    Expectation Maximisation (EM) algorithm for probabilistic inversion of
    thermohydraulic correlation, with diagonal covariance matrix.
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

        # Cast to NumPy arrays for consistency
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

        # Initial mean and covariance (full diagonal matrix for public API)
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

        # Internal 1D representations (diagonal covariance as a vector)
        self._mean0 = self.initial_mean.reshape(-1).astype(float)
        self._gamma0 = np.diag(self.initial_cov).astype(float)

    @staticmethod
    def _one_step_em_diag(mean, gamma, h, z_exp, z_nom, sig_eps):
        """
        One EM step using only the diagonal of the covariance (gamma).

        Parameters
        ----------
        mean  : (p,)
        gamma : (p,)
        h     : (p, n)
        """

        # Residual for each observation: z_exp - z_nom - h_i^T mean
        resid = z_exp - z_nom - np.sum(h * mean[:, None], axis=0)  # (n,)

        # Denominators: h_i^T cov h_i + sig_eps^2, with diagonal cov
        denom = sig_eps**2 + np.sum(gamma[:, None] * h**2, axis=0)  # (n,)

        # g_ji = (cov @ h_i)_j = gamma_j * h_ji
        g = gamma[:, None] * h  # (p, n)

        # b_ij and S_ij for diagonal update
        b = g * resid[None, :] / denom[None, :]  # (p, n)
        S_diag = (g**2) / denom[None, :]  # (p, n)
        delta_cov_diag_mat = b**2 - S_diag  # (p, n)

        # Empirical means over observations
        delta_mean = np.mean(b, axis=1)  # (p,)
        delta_cov_diag = np.mean(delta_cov_diag_mat, axis=1)  # (p,)

        mean_new = mean + delta_mean
        gamma_new = gamma + delta_cov_diag - delta_mean**2

        return mean_new, gamma_new

    @staticmethod
    def _loglik_diag(mean, gamma, h, z_exp, z_nom, sig_eps):
        """
        Log-likelihood with diagonal covariance (gamma).
        """
        resid = z_exp - z_nom - np.sum(h * mean[:, None], axis=0)  # (n,)
        denom = sig_eps**2 + np.sum(gamma[:, None] * h**2, axis=0)  # (n,)

        loglik = -0.5 * np.sum(resid**2 / denom) + 0.5 * np.sum(np.log(denom))
        return np.array([loglik])

    def _run_single_em(self, init_mu, init_gamma):
        """
        Run one EM trajectory to convergence from a given starting point.

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
            )[0]
        ]

        mean, gamma = init_mu, init_gamma
        iterator = 0
        err_cov = np.inf
        err_mean = np.inf

        while (err_cov > self.tolerance or err_mean > self.tolerance) and (
            self.niter is None or iterator < self.niter
        ):
            mean_new, gamma_new = self._one_step_em_diag(
                mean, gamma, self.h, self.z_exp, self.z_nom, self.sig_eps
            )

            cov_list.append(np.diag(gamma_new))
            mean_list.append(mean_new.reshape(-1, 1))
            loglik_list.append(
                self._loglik_diag(
                    mean_new, gamma_new, self.h, self.z_exp, self.z_nom, self.sig_eps
                )[0]
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
        Run the EM iterations until convergence (or until niter is reached),
        restarting from multiple initial points and keeping the run with the
        highest final log-likelihood.

        Parameters
        ----------
        n_starts : int
            Number of starting points. The first start always uses the
            (initial_mean, initial_cov) provided at construction; the
            remaining starts are drawn at random around it.
        random_state : int or numpy.random.Generator, optional
            Seed or generator used for the random restarts, for reproducibility.

        Returns
        -------
        mean_list : list of ndarray
            Sequence of mean vectors for the best run.
        cov_list : list of ndarray
            Sequence of (diagonal) covariance matrices for the best run.
        loglik_list : list of float
            Log-likelihood values at each iteration for the best run
            (same length as mean_list/cov_list).
        iterator : int
            Number of EM steps taken for the best run.
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

            mean_list, cov_list, loglik_list, iterator = self._run_single_em(
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
                "EM failed to converge to a finite log-likelihood from any of the "
                f"{n_starts} starting points."
            )

        mean_list, cov_list, loglik_list, iterator = best_res

        return mean_list, cov_list, loglik_list, iterator
