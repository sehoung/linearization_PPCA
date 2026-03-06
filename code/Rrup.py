import numpy as np
import numba
from numba.typed import List


# --- WGS-84 Ellipsoid Constants ---
# Used for converting geodetic coordinates (lat, lon, alt) to Cartesian (x, y, z)
A = 6378137.0  # Semi-major axis (equatorial radius) in meters
F = 1 / 298.257223563  # Flattening
E_SQ = F * (2 - F)  # Eccentricity squared


@numba.jit(nopython=True)
def calc_Rrup_numba(rupture_coord_list, site_coord):
    """
    Calculates the distance from each point in an array to a finite rectangle.
    All inputs are expected to be in ECEF Cartesian coordinates.

    Args:
        rupture_coord_list : (Nobs, 4, 3) Stack of (4,3) rupture coordinates in la, lo, elevation (m)
        site_coord (np.array): (Nb, 3), columns are la, lo, elevation (m)

    Returns:
        np.array: (Nobs, Nb), distances in km
    """

    
    site_ecef = geodetic_to_ecef(site_coord)

    num_points = site_ecef.shape[0]
    distances = np.empty(num_points, dtype=np.float64)

    dist_list = List()

    for rup in rupture_coord_list:
        rup_ecef = geodetic_to_ecef(rup)
        for i in range(num_points):
            point = site_ecef[i]
            distances[i] = distance_point_to_rupture(rup_ecef[0], rup_ecef[1], rup_ecef[2], rup_ecef[3], point)*0.001
        dist_list.append(distances.copy())
    #dist_list = np.array(dist_list)
    return dist_list





@numba.jit(nopython=True)
def geodetic_to_ecef(geo_points_array):
    """
    Converts an array of Geodetic coordinates to an array of ECEF coordinates.

    Args:
        geo_points_array (np.array): A 2D numpy array where each row is a point
                                     [latitude, longitude, altitude].

    Returns:
        np.array: A 2D numpy array where each row is the corresponding
                  ECEF coordinate [x, y, z] in meters.
    """
    num_points = geo_points_array.shape[0]
    ecef_points_array = np.empty_like(geo_points_array)
    
    for i in range(num_points):
        lat, lon, alt = geo_points_array[i]

        # Convert latitude and longitude from degrees to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Calculate the radius of curvature in the prime vertical
        n = A / np.sqrt(1 - E_SQ * np.sin(lat_rad)**2)
        
        # Calculate ECEF coordinates
        x = (n + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (n + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (n * (1 - E_SQ) + alt) * np.sin(lat_rad)
        
        ecef_points_array[i] = np.array([x, y, z])
        
    return ecef_points_array




@numba.jit(nopython=True)
def distance_point_to_rupture(r1, r2, r3, r4, point):
    """
    Calculates the shortest distance from a 3D point to a finite rupture.
    
    The rectangle is defined by four corner vertices, given in order (e.g., counter-clockwise).
    This implementation assumes r1, r2, and r4 define the corner and adjacent sides.

    Args:
        r1, r2, r3, r4 (np.array): Four points defining the rectangle's corners.
        point (np.array): The point to calculate the distance from.

    Returns:
        float: The shortest distance from the point to the rectangle.
    """
    # 1. Define the rectangle's local coordinate system vectors from r1.
    u = r2 - r1  # Local x-axis
    v = r4 - r1  # Local y-axis
    w = point - r1

    # 2. Project the point's vector (w) onto the local axes
    u_dot_u = np.dot(u, u)
    v_dot_v = np.dot(v, v)

    if u_dot_u < 1e-9 or v_dot_v < 1e-9:
        return np.inf

    alpha = np.dot(w, u) / u_dot_u
    beta = np.dot(w, v) / v_dot_v

    # 3. Clamp the coordinates to the bounds of the rectangle [0, 1].
    alpha_clamped = max(0.0, min(1.0, alpha))
    beta_clamped = max(0.0, min(1.0, beta))

    # 4. Find the closest point on the rectangle
    closest_point_on_rect = r1 + alpha_clamped * u + beta_clamped * v
    
    # 5. Calculate the Euclidean distance
    dist_vec = point - closest_point_on_rect
    distance = np.sqrt(np.dot(dist_vec, dist_vec))
    
    return distance


