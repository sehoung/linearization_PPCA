
import numpy as np
from scipy.interpolate import RectBivariateSpline
import itertools as it

from numba import njit, prange

# --- STEP 1: Define constants (do this ONCE at the top of your script) ---
# Tlist and B-matrices from your code
Tlist = np.array([0.01, 0.1, 0.2, 0.5, 1, 2, 5, 7.5, 10])

B1 = np.array([
    [0.29, 0.25, 0.23, 0.23, 0.18, 0.10, 0.06, 0.06, 0.06],
    [0.25, 0.30, 0.20, 0.16, 0.10, 0.04, 0.03, 0.04, 0.05],
    [0.23, 0.20, 0.27, 0.18, 0.10, 0.03, 0.00, 0.01, 0.02],
    [0.23, 0.16, 0.18, 0.31, 0.22, 0.14, 0.08, 0.07, 0.07],
    [0.18, 0.10, 0.10, 0.22, 0.33, 0.24, 0.16, 0.13, 0.12],
    [0.10, 0.04, 0.03, 0.14, 0.24, 0.33, 0.26, 0.21, 0.19],
    [0.06, 0.03, 0.00, 0.08, 0.16, 0.26, 0.37, 0.30, 0.26],
    [0.06, 0.04, 0.01, 0.07, 0.13, 0.21, 0.30, 0.28, 0.24],
    [0.06, 0.05, 0.02, 0.07, 0.12, 0.19, 0.26, 0.24, 0.23],
], dtype=float)

B2 = np.array([
    [0.47, 0.40, 0.43, 0.35, 0.27, 0.15, 0.13, 0.09, 0.12],
    [0.40, 0.42, 0.37, 0.25, 0.15, 0.03, 0.04, 0.00, 0.03],
    [0.43, 0.37, 0.45, 0.36, 0.26, 0.15, 0.09, 0.05, 0.08],
    [0.35, 0.25, 0.36, 0.42, 0.37, 0.29, 0.20, 0.16, 0.16],
    [0.27, 0.15, 0.26, 0.37, 0.48, 0.41, 0.26, 0.21, 0.21],
    [0.15, 0.03, 0.15, 0.29, 0.41, 0.55, 0.37, 0.33, 0.32],
    [0.13, 0.04, 0.09, 0.20, 0.26, 0.37, 0.51, 0.49, 0.49],
    [0.09, 0.00, 0.05, 0.16, 0.21, 0.33, 0.49, 0.62, 0.60],
    [0.12, 0.03, 0.08, 0.16, 0.21, 0.32, 0.49, 0.60, 0.68],
], dtype=float)

B3 = np.array([
    [ 0.24,  0.22,  0.21,  0.09, -0.02,  0.01,  0.03,  0.02,  0.01],
    [ 0.22,  0.28,  0.20,  0.04, -0.05,  0.00,  0.01,  0.01, -0.01],
    [ 0.21,  0.20,  0.28,  0.05, -0.06,  0.00,  0.04,  0.03,  0.01],
    [ 0.09,  0.04,  0.05,  0.26,  0.14,  0.05,  0.05,  0.05,  0.04],
    [-0.02, -0.05, -0.06,  0.14,  0.20,  0.07,  0.05,  0.05,  0.05],
    [ 0.01,  0.00,  0.00,  0.05,  0.07,  0.12,  0.08,  0.07,  0.06],
    [ 0.03,  0.01,  0.04,  0.05,  0.05,  0.08,  0.12,  0.10,  0.08],
    [ 0.02,  0.01,  0.03,  0.05,  0.05,  0.07,  0.10,  0.10,  0.09],
    [ 0.01, -0.01,  0.01,  0.04,  0.05,  0.06,  0.08,  0.09,  0.09],
], dtype=float)


# --- STEP 2: Create interpolators (do this ONCE at the top of your script) ---
# kx=1, ky=1 selects fast linear interpolation
RBS1 = RectBivariateSpline(Tlist, Tlist, B1, kx=1, ky=1)
RBS2 = RectBivariateSpline(Tlist, Tlist, B2, kx=1, ky=1)
RBS3 = RectBivariateSpline(Tlist, Tlist, B3, kx=1, ky=1)



def coord_to_corr_mat_loth2013(coord, period = 0.01, cholesky=True):
    
    # Calculate the pairwise distance from the coordinates
    h = _coord_to_dist_numba(coord)
    

    #  Get the coefficients for the correlation calculation
    b1 = RBS1.ev(period, period).item()
    b2 = RBS2.ev(period, period).item()
    b3 = RBS3.ev(period, period).item()

    # calculate the correlations
    rho = _getCorrelation_numba(h, b1, b2, b3)
    del h
    rho_max = np.max(rho)
    
    # Convert the N(N-1)/2 1-d array to N by N symmetric correlation matrix
    N = len(coord)
    C = _list_to_symm_matrix(rho, N, fill_diagonal = 1)
    del rho
    #rho_max = np.triu(C, k=1).max()
    C = C + 1e-6*np.eye(N) # regularization for stability
    
    # Conduct Cholesky decomposition
    if cholesky == True:
        L = np.linalg.cholesky(C) 
    else:
        L = None

    return C, L, rho_max



@njit
def _coord_to_dist_numba(coords):
    """
    Compute the N_upper-sized 1D array of Haversine distances (km) efficiently
    using Numba-compiled loops.

    Parameters
    ----------
    coords : ndarray of shape (N, 2)
        Each row is [latitude, longitude] in decimal degrees.
        Must be a NumPy array for Numba.

    Returns
    -------
    dist_array : ndarray of shape (N_upper,)
        1D array of great-circle distances in kilometers for the upper triangle.
        N_upper = N * (N - 1) / 2
    """
    R = 6371.0  # Earth radius in km
    N = coords.shape[0]

    # --- Pre-allocate 1D output array ---
    # The number of pairs in the upper triangle (k=1) is N*(N-1)/2
    n_upper = (N * (N - 1)) // 2
    dist_array = np.empty(n_upper, dtype=np.float64)
    
    # --- Convert all coordinates to radians once ---
    coords_rad = np.radians(coords)
    
    # Precompute all cos(lat) terms
    cos_lat = np.cos(coords_rad[:, 0])

    # Index for the 1D output array
    k = 0
    
    # --- Iterate over the upper triangle ---
    for i in range(N):
        # Get values for point i
        lat1 = coords_rad[i, 0]
        lon1 = coords_rad[i, 1]
        cos_lat1 = cos_lat[i]
        
        # Start j from i+1 to only compute the upper triangle
        for j in range(i + 1, N):
            
            # Get values for point j
            lat2 = coords_rad[j, 0]
            lon2 = coords_rad[j, 1]
            cos_lat2 = cos_lat[j]
            
            # --- Haversine formula ---
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            sin_dlat2 = np.sin(dlat * 0.5)
            sin_dlon2 = np.sin(dlon * 0.5)
            
            a = sin_dlat2**2 + (cos_lat1 * cos_lat2) * (sin_dlon2**2)
            
            # Clip 'a' to [0, 1] for numerical stability
            if a < 0.0:
                a = 0.0
            elif a > 1.0:
                a = 1.0
                
            c = 2.0 * np.arcsin(np.sqrt(a))
            
            # Store the result
            dist_array[k] = R * c
            k += 1 # Increment 1D array index
            
    return dist_array


@njit(fastmath=True)
def _getCorrelation_numba(h, b1, b2, b3):
    """
    Calculates the correlation based on the Loth2013 model,
    optimized with Numba.
    
    Parameters
    ----------
    h : ndarray
        Array of distances (can be 1D or 2D).
    b1, b2, b3 : float
        Model coefficients.

    Returns
    -------
    rho : ndarray
        Correlation array, same shape as h.
    """
    # Create an output array of the same shape as h
    rho = np.empty(h.shape, dtype=np.float64)
    
    # Iterate over every element in h.
    # Numba's @njit will optimize this loop.
    # We use .flat to iterate over all elements regardless
    # of whether h is 1D or 2D.
    for i in range(h.size):
        dist = h.flat[i]
        
        # Calculate the two exponential terms
        term1 = b1 * np.exp(-3.0 * dist / 20.0)
        term2 = b2 * np.exp(-3.0 * dist / 70.0)
        
        # Handle the (h == 0) * b3 part
        # This is much faster than array-based boolean checks
        if dist == 0.0:
            term3 = b3
        else:
            term3 = 0.0
            
        rho.flat[i] = term1 + term2 + term3
        
    return rho



def _list_to_symm_matrix(d, N, fill_diagonal = 1):
    # Fill symmetric matrix
    # N = N of coordiates
    # E.G.>
    # If list is distance matrix, fill_diagonal = 0
    # If list is correlation matrix, fill_diagonal = 1
    #N = d.size

    D = np.full((N, N), fill_diagonal, dtype=float)
    iu, ju = np.triu_indices(N, k=1)
    D[iu, ju] = d
    D[ju, iu] = d
    return D

