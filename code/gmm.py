import numpy as np
import numba


def GMM_mean_over_Nobs(M, R):
    """
    M = (Nobs,)
    R = (Nobs, Nb)
    """
    # Initialize coefficients arrays
    C1 = np.zeros_like(M)
    C2 = np.zeros_like(M)
    C3 = np.zeros_like(M)
    C4 = -2.100 * np.ones_like(M)
    C5 = np.zeros_like(M)
    C6 = np.zeros_like(M)
    C7 = np.zeros_like(M)

    # Conditions based on M values
    condition = M <= 6.5
    C1[condition] = -0.624
    C2[condition] = 1.0
    C3[condition] = 0.000
    C5[condition] = 1.29649
    C6[condition] = 0.250

    condition = M >= 6.5
    C1[condition] = -1.274
    C2[condition] = 1.1
    C3[condition] = 0.000
    C5[condition] = -0.48451
    C6[condition] = 0.524

    # Calculate lnmean for all M and R
    Rrup = R
    f_mag = C1 + C2 * M + C3 * (8.5 * M)**2.5
    f_geo = C4[:,None] * np.log(Rrup + np.exp(C5 + C6 * M)[:,None])
    f_anela = C7[:,None] * np.log(Rrup + 2)
    lnmean = f_mag[:,None] + f_geo +  f_anela

    return lnmean


@numba.njit
def GMM_stdev_over_Nobs(M):
    """
    M = (Nobs, Nb)
    """

    M_broadcasted = M
    lnsigma = np.where(M_broadcasted < 7.21, 1.39 - 0.14 * M_broadcasted, 0.38)
    lntau = np.where(M_broadcasted < 10.0, 0.2, 0.2)  # constant here
    lnphi = np.sqrt(lnsigma**2 - lntau**2)

    return lntau, lnphi