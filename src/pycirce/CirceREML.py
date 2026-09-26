#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagonal-covariance restricted likelihood optimisation for probabilistic inversion of
thermohydraulic correlations (pure Python, vectorised NumPy).
"""

import numpy as np


class CirceREML:
    """
    Restricted maximum likelihood algorithm for probabilistic inversion of
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
    def restricted_nloglik_diag(gamma, h, y, sig_eps):
        """
        Negative restricted log-likelihood with diagonal covariance of the latent variable (gamma).
        """
        p, n = h.shape

        # Residual with mean=0 (initial approx for w calculation)
        # However, REML NLL formula is: 0.5 * log|V| + 0.5 * r^T V^-1 r + 0.5 * log|X^T V^-1 X| - const
        # where r = z_exp - z_nom - H * beta_hat
        # beta_hat = (H^T V^-1 H)^-1 H^T V^-1 (z_exp - z_nom)
        # V = diag(sig_eps^2 + sum(gamma * h^2))

        # 1. Compute diagonal of V
        # Gamma must be positive, enforced by bounds in estimate()
        denom = sig_eps**2 + np.sum(gamma[:, None] * h**2, axis=0)  # (n,) diagonal of V
        w = 1.0 / denom  # (n,) diagonal of V^-1

        # H_tilde = H^T V^-1 H = sum_i w_i h_i h_i^T
        H_tilde = (h * w[None, :]) @ h.T  # (p, p)

        # Moore–Penrose inverse of H_tilde via SVD for stability
        U, s, Vt = np.linalg.svd(H_tilde)
        s_pseudo = np.zeros_like(s)
        non_zero = s > 1e-10
        s_pseudo[non_zero] = 1.0 / s[non_zero]
        H_plus = Vt.T @ np.diag(s_pseudo) @ U.T

        P = np.diag(w) - (h * w[None, :]).T @ H_plus @ (h * w[None, :])

        # 4. Compute NLL terms
        # Term 1: log|V| = sum log(denom)
        log_det_V = np.sum(np.log(denom))

        # Term 2: r^T V^-1 r = sum resid_i^2 / denom_i
        quad_form = y.dot(P @ y.T)

        # Term 3: log|H^T V^-1 H| = log|H_tilde|
        # Use pseudo-determinant (product of non-zero singular values)
        log_det_H_tilde = np.sum(np.log(s[non_zero]))

        nll = log_det_V + quad_form + log_det_H_tilde
        return nll

    def estimate(self, n_starts=5):
        """
        Run the REML estimation using numerical optimization (Scipy) with multistart.

        Parameters
        ----------
        n_starts : int
            Number of starting points for the optimization (default=5).

        Returns
        -------
        mean_list : list of ndarray
            Sequence of mean vectors (last one is the final estimate).
        cov_list : list of ndarray
            Sequence of (diagonal) covariance matrices.
        loglik_list : list of float
            Restricted log-likelihood values (negative of NLL).
        err_cov_list : list of float
            Not used for direct optimization, returns empty or final error.
        err_mean_list : list of float
            Not used for direct optimization, returns empty or final error.
        """
        from scipy.optimize import minimize

        # Bounds for gamma (variance must be non-negative)
        # We use a small epsilon to avoid division by zero if sig_eps is 0
        p = self.h.shape[0]
        bounds = [(1e-10, None) for _ in range(p)]
        y = self.z_exp - self.z_nom

        # Objective function wrapper
        def objective(gamma_current):
            return self.restricted_nloglik_diag(
                gamma_current, self.h, y, self.sig_eps
            )

        best_fun = np.inf
        best_res = None
        
        # Try multiple starting points
        for i in range(n_starts):
            if i == 0:
                init_gamma = self._gamma0
            else:
                # Random initialization: scale initial guess or uniform
                # Here we sample around the initial guess or just random positive
                # Let's assume initial_cov gives the scale
                scale = np.mean(self._gamma0) if np.sum(self._gamma0) > 1e-9 else 1.0
                init_gamma = np.random.uniform(0.1 * scale, 10 * scale, size=p)

            res = minimize(
                objective,
                init_gamma,
                method="L-BFGS-B",
                bounds=bounds,
                tol=self.tolerance,
                options={"maxiter": self.niter if self.niter else 15000}
            )
            
            if res.fun < best_fun:
                best_fun = res.fun
                best_res = res
        
        res = best_res
        final_gamma = res.x
        final_nll = res.fun

        # Recompute final mean using closed form GLS
        denom = self.sig_eps**2 + np.sum(final_gamma[:, None] * self.h**2, axis=0)
        w = 1.0 / denom
        H_tilde = (self.h * w[None, :]) @ self.h.T
        
        U, s, Vt = np.linalg.svd(H_tilde)
        s_pseudo = np.zeros_like(s)
        non_zero = s > 1e-10
        s_pseudo[non_zero] = 1.0 / s[non_zero]
        H_plus = Vt.T @ np.diag(s_pseudo) @ U.T
        
        y = self.z_exp - self.z_nom
        rhs = self.h @ (y * w)
        final_mean = H_plus @ rhs

        # Loglik list: usually increasing, so return negative of min NLL
        loglik_list = [-objective(self._gamma0), -final_nll]
        

        return final_mean, final_gamma, loglik_list