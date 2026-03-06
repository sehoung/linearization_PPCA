# import os
# os.environ["OMP_NUM_THREADS"] = "1"          # OpenMP
# os.environ["OPENBLAS_NUM_THREADS"] = "1"     # OpenBLAS
# os.environ["MKL_NUM_THREADS"] = "1"          # Intel MKL
# os.environ["BLIS_NUM_THREADS"] = "1"         # BLIS
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # Apple Accelerate
# os.environ["NUMEXPR_NUM_THREADS"] = "1"      # numexpr (if you use it)


from PPCA_batch import PPCA_batch
from misc_func import cleanup_ppca_files
if __name__ =="__main__":
    ## Set building inventory ##
    df_fname = "./data/SanFrancisco_buildings_ext.csv"
    #------------------------#

    ## Set Nb, Nobs, tmax ##
    Nb_list =  [1000, 2000, 5000] # Sample the first 1000 buildings from the repository
    Nobs_list = [10_000] # [1_000_000] #, 32000, 64000, 128000, 256000, 512000]
    tmax_list = [1, 2, 5, 10]

    cleanup_ppca_files()    
    PPCA_batch(df_fname, Nb_list, Nobs_list, tmax_list)

