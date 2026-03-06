import numpy as np
import numba
import math

@numba.njit
def calc_g_sim_ds(eps, lnY, theta, beta):
    """
    eps (Nobs, Nb)
    lnY (Nobs, Nb)
    theta(Nb, Nds)
    beta(Nb, Nds)
    """
    Nb = len(theta)
    Nobs = len(lnY)
    Nds = np.shape(theta)[1]  
    
    ds = np.empty((Nobs, Nb), dtype=np.int8)
    log_theta = np.log(theta)
    
    # Simulate ds
    for i_obs in range(Nobs):
        for i_b in range(Nb):
            eps_ij = eps[i_obs, i_b]
            lnY_ij = lnY[i_obs, i_b]
            
            ds_cur=0
            for i_ds in range(Nds):

                log_theta_jk = log_theta[i_b, i_ds]
                beta_jk = beta[i_b, i_ds]

                pf = norm_cdf_numba( (lnY_ij - log_theta_jk) / beta_jk )
                g_ijk = eps_ij  - pf


                if g_ijk>0:
                    ds_cur=i_ds
                    break
                else:
                    ds_cur+=1
            ds[i_obs, i_b] = ds_cur
    return ds


@numba.njit
def norm_cdf_numba(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


