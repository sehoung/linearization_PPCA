import numpy as np
import time
import pandas as pd
import numba
from scipy.sparse.linalg import eigsh
# from dataclasses import dataclass

# @dataclass
# class out_ds:
#     t: pd.DataFrame
#     ds: np.ndarray
#     sv: np.ndarray
#     sv2: np.ndarray



def construct_AAT_for_one_Nobs(
    tau: np.ndarray,        # (N_asset, )
    phi: np.ndarray,        # (N_asset, )
    C: np.ndarray,          # (N_asset, N_asset)
    beta: np.ndarray,       # (N_asset, N_ds)
) -> np.ndarray:
    """Computes a stacked 3D array of An matrices using vectorization."""
    N_asset = len(tau)
    I = N_asset

    beta_k = beta[:, 0]
    inv_beta = 1.0 / beta_k # Shape: (I,)
    
    #D_inv_beta = np.diag(inv_beta)
    #D_phi = np.diag(phi)
    inv_beta_phi = inv_beta*phi
    #D_inv_beta_phi = np.diag(inv_beta_phi)
    term1 = np.eye(N_asset)
    term2 = np.diag(inv_beta*inv_beta) @(tau*tau)     # (D_inv_beta*D_inv_beta) @ (tau*tau)
    
    #D_inv_beta_phi = np.diag(inv_beta_phi)
    term3 = inv_beta_phi * C * inv_beta_phi
    #term3 = D_inv_beta @ D_phi @ C @ D_phi @ D_inv_beta

    AAT = term1 + term2 + term3

    return AAT
    


def step02_calc_c_sigma(rho_max, phi, beta_full):
    c_sigma = ( 1 + (1-rho_max) * phi[0,0]**2 / beta_full[0,0]**2)**0.5
    return c_sigma


def step03_construct_Wt(AAt, c_sigma, tmax=None):

    if tmax == None:
        t_svd = time.time()
        #s_all = None
        s_denoise = None
        t_svd = time.time() - t_svd
    else:
        t_svd = time.time()
        s_all_sq, U = eigsh(AAt, k=tmax, which='LM')
        s_sq_denoise = s_all_sq - c_sigma**2
        s_denoise = ( np.where(s_sq_denoise <0, 0, s_sq_denoise) )**0.5 # To prevent negative sqrt due to the numerical error
        Wt = U @ np.diag(s_denoise)
        t_svd = time.time() - t_svd

    return Wt

## Sample random variables
def step04_sampling(Nobs, Nb, tmax):
    if tmax == 1:
        X = np.random.standard_normal((Nobs, tmax))
        noise = np.random.standard_normal((Nobs, Nb))
        # Reshape X if you need the (-1, 1) final shape, otherwise X is (Nobs, 1)
        X = X.reshape(-1, 1)
    elif tmax is not None:
        X = np.random.standard_normal((Nobs, tmax))
        noise = np.random.standard_normal((Nobs, Nb))
    else: # tmax is None
        X = np.random.standard_normal((Nobs, 2 * Nb + 1))
        noise = np.zeros((Nobs, Nb))

    return X, noise

@numba.njit
def step05_calc_g_sim_ds(Wt, X, c_sigma, noise, mu, beta, theta):
    """
    Wt (Nb, tmax)
    X (Nobs, t max)
    c_sigma (float)
    noise (Nobs, Nb)
    mu (Nobs, Nb)
    beta (Nb, Nds)
    theta (Nb, Nds)

    g (Nobs, Nb)
    """
    Nb = len(Wt)
    Nobs = len(X)
    Nds = np.shape(theta)[1]  
    g = np.empty((Nobs, Nb))
    ds = np.empty((Nobs, Nb), dtype=np.int8)
    inv_beta = 1/beta

    log_theta = np.log(theta)
    #mu_div_beta = mu*inv_beta[:,]


    # g before adding bk
    for i_obs in range(Nobs):
        g[i_obs,:] = Wt @ X[i_obs,:] + c_sigma*noise[i_obs,:]  # (Nb, )
    
    # add g to bk and simulate ds
    for i_obs in range(Nobs):
        for i_b in range(Nb):
            g_i_obs_i_b = g[i_obs,i_b]
            mu_i_obs_i_b = mu[i_obs,i_b]
            
            ds_cur=0
            for i_ds in range(Nds):
                g_i_obs_i_b_i_ds = g_i_obs_i_b + (log_theta[i_b,i_ds] - mu_i_obs_i_b) * inv_beta[i_b,i_ds]
                if g_i_obs_i_b_i_ds>0:
                    ds_cur=i_ds
                    break
                else:
                    ds_cur+=1
            ds[i_obs, i_b] = ds_cur
    return ds

