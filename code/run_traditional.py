import os 
os.environ["OMP_NUM_THREADS"] = "1"          # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"     # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"          # Intel MKL
os.environ["BLIS_NUM_THREADS"] = "1"         # BLIS
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # Apple Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"      # numexpr (if you use it)

import pandas as pd
import time
import itertools
import numpy as np
import gc
from numpy.lib.format import open_memmap

from GMcorrelation import coord_to_corr_mat_loth2013
from gmm import GMM_mean_over_Nobs, GMM_stdev_over_Nobs
from Rrup import calc_Rrup_numba
from mapping_df import match_fragility, match_loss_ratio, match_repair_time
from ds_trad_determ import calc_g_sim_ds
from loss import ds_to_repairtime, ds_to_loss

from misc_func import cleanup_trad_files

import psutil
import os

def print_memory(label=""):
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss # Resident Set Size
    mem_mb = mem_bytes / (1024 ** 2)      # Convert to MB
    print(f"[MEMORY] {label}: {mem_mb:.2f} MB")



if __name__ == "__main__":

    cleanup_trad_files()    

    df_fname = "./data/SanFrancisco_buildings_ext.csv"

    # Select assets #
    Nb_list =   [1000, 2000, 5000]
    Nobs_list = [10_000]
    Nb_Nobs_combination = list(itertools.product(Nb_list, Nobs_list))
    BATCH_SIZE = 10_000

    df_comptime = pd.DataFrame(Nb_Nobs_combination, columns = ["Nb", "Nobs"])

    ## Output lookup Table ##
    outdir = "./out/"
    outdir_npy = outdir+"/npy"
    outdir_df = outdir+"/selected_buildings"
    
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_npy, exist_ok=True)
    os.makedirs(outdir_df, exist_ok=True)
    fname_lookup = outdir+'trad_lookup.csv'
    df_lookup = pd.DataFrame(Nb_Nobs_combination, columns = ["Nb", "Nobs"])
    df_lookup['i_case'] = range(len(Nb_Nobs_combination))
    df_lookup.to_csv(fname_lookup, index=False)

    
    #----------------------------#


    
    i_case = 0
    for Nb in Nb_list:

        id_sel = range(0, Nb)
        df = pd.read_csv(df_fname)
        df_sel = df[df["id"].isin(id_sel)]

        ## Match the fragility function ##
        match_frag_fname = "./data/fragility_PGA.csv"
        df_sel, theta, beta = match_fragility(df_sel, mapping_fname=match_frag_fname)
        Nds = theta.shape[1]       
        #--------------------------#

        ## Match loss ratio##
        match_loss_fname = "./data/consequence_repair_PGA.csv"
        df_sel, lossratio_ds = match_loss_ratio(df_sel, mapping_fname=match_loss_fname)
        #---------------------------------#

        ## Match repair time##
        match_repair_fname = "./data/consequence_repair_PGA.csv"
        df_sel, repairtime_ds = match_repair_time(df_sel, mapping_fname=match_repair_fname)
        #--------------------------#
        
        ## Get cost of buildings #
        cost = df_sel['ReplacementCost'].to_numpy()
        #----------------#

        df_outname = "trad_" + (df_fname.split('/')[-1]).split('.')[0] +  f"_{i_case}" + ".csv"
        df_sel.to_csv(f"{outdir_df}/{df_outname}", index=False)
        #df_sel.to_csv(df_outname, index=False)
        Nb = len(df_sel)

        asset_coord = np.c_[df_sel['Latitude'], df_sel['Longitude'], np.full(Nb, 0.0)]

        # Correlation Matrix #
        t= time.time()
        C, L, rho_max = coord_to_corr_mat_loth2013(asset_coord, cholesky=False) # This may be the expensive, L = (Nb, Nb)
        t_corr = time.time() - t
       
        t=time.time()
        L = np.linalg.cholesky(C)
        del C
        t_cholesky = time.time()-t

        for ii in range(len(Nobs_list)):
            df_comptime.loc[i_case+ii, 't_corr'] = t_corr
            df_comptime.loc[i_case+ii, 't_cholesky'] = t_cholesky

        

        ### Earthquake scenario setting ###
        Mag_list = np.full(1, 7.2)
        rup_coord = np.array([[37.7776, -122.5558, 0],
                                [37.3604, -122.3447, 0],
                                [37.3604, -122.3447, 10],
                                [37.7776, -122.5558, 10]])
        rup_coord_list = np.tile(rup_coord, (1, 1, 1)) 
        #--------------------------------------#

        ## Calc Rrup, mu, and stdev for the single scenario ##
        calc_Rrup_numba(rup_coord_list[:2], asset_coord[:2]) # First run just for compiling numba function
        t=time.time()
        Rrup_list = calc_Rrup_numba(rup_coord_list, asset_coord) # Rrup2 = (Nobs, sNb)
        Rrup_list = np.array(Rrup_list)
        
        for ii in range(len(Nobs_list)):
            df_comptime.loc[i_case+ii, 't_Rrup'] = time.time() - t
        
        GMM_mean_over_Nobs(Mag_list[:2], Rrup_list[:2,:2]) # First compile
        t=time.time()
        mu_single = GMM_mean_over_Nobs(Mag_list, Rrup_list) # input = M (Nobs,); R (Nobs,Nb)
        
        for ii in range(len(Nobs_list)):
            df_comptime.loc[i_case+ii, 't_mu'] = time.time() - t

        t=time.time()
        tau_single, phi_single = GMM_stdev_over_Nobs(Mag_list)
        
        for ii in range(len(Nobs_list)):
            df_comptime.loc[i_case+ii, 't_stdev'] = time.time() - t



        for i_Nobs, Nobs in enumerate(Nobs_list):
            tau = tau_single[0]
            phi = phi_single[0]
            # Note: We do NOT tile 'mu' for the full Nobs yet to save memory.
            
            # 2. Initialize Timers to 0.0 (since we will accumulate them per batch)
            df_comptime.loc[i_case, 't_smpl']    = 0.0
            df_comptime.loc[i_case, 't_GMsim']   = 0.0
            df_comptime.loc[i_case, 't_g_ds']    = 0.0
            df_comptime.loc[i_case, 't_loss']    = 0.0
            df_comptime.loc[i_case, 't_repair']  = 0.0

            # 3. Initialize RNG *ONCE* here to ensure global consistency
            # (Do not put this inside the batch loop, or you restart the random sequence!)
            rng = np.random.default_rng(seed=42) 

            # 4. Prepare Output Filenames
            outfname_ds = f"./out/npy/trad_ds_{i_case}_.npy"
            outfname_loss = f"./out/npy/trad_loss_{i_case}_.npy"
            outfname_repairtime = f"./out/npy/trad_repairtime_{i_case}_.npy"

            # --- STEP 0: PRE-ALLOCATE DISK SPACE ---
            # We create the files once, then close them.
            # This allows us to use mode='r+' (modify) inside the loop safely.
            # We enforce float32 to save 50% space.
            
            mm = open_memmap(outfname_ds, mode='w+', dtype=np.int8, shape=(Nobs, Nb))
            del mm
            mm = open_memmap(outfname_loss, mode='w+', dtype=np.float64, shape=(Nobs, Nb))
            del mm
            mm = open_memmap(outfname_repairtime, mode='w+', dtype=np.float64, shape=(Nobs, Nb))
            del mm

            # --- JIT WARMUP (Optional but recommended) ---
            # Run a dummy pass on tiny data to compile Numba functions before timing
            _dum_eps = rng.uniform(size=(2, Nb))
            _dum_lnY = rng.normal(size=(2, Nb))
            _ = calc_g_sim_ds(_dum_eps, _dum_lnY, theta, beta)

            # 5. Start Batch Loop
            print(f"Nb={Nb}: Processing Nobs={Nobs} in batches of {BATCH_SIZE}...")
            
            for start_idx in range(0, Nobs, BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, Nobs)
                current_nobs = end_idx - start_idx
                
                # --- A. Sampling ---
                t = time.time()
                # Generate ONLY enough data for this batch
                zeta_batch = rng.normal(size=current_nobs)
                Xi_batch   = rng.normal(size=(current_nobs, Nb))
                eps_batch  = rng.uniform(size=(current_nobs, Nb))
                df_comptime.loc[i_case, 't_smpl'] += time.time() - t

                # --- B. GM Simulation ---
                t = time.time()
                # Tile mu only for this small batch
                mu_batch = np.tile(mu_single, (current_nobs, 1)) 
                lnY_batch = mu_batch + tau * zeta_batch[:, None] + phi * (Xi_batch @ L.T)
                df_comptime.loc[i_case, 't_GMsim'] += time.time() - t
                
                # --- C. Limit-State / DS ---
                t = time.time()
                ds_batch = calc_g_sim_ds(eps_batch, lnY_batch, theta, beta)
                df_comptime.loc[i_case, 't_g_ds'] += time.time() - t

                # --- D. Loss ---
                t = time.time()
                loss_batch = ds_to_loss(ds_batch, lossratio_ds, cost)
                df_comptime.loc[i_case, 't_loss'] += time.time() - t

                # --- E. Repair Time ---
                t = time.time()
                repair_batch = ds_to_repairtime(ds_batch, repairtime_ds)
                df_comptime.loc[i_case, 't_repair'] += time.time() - t


                # --- F. Save to Disk (WINDOWED WRITE) ---
                # Open ONLY the slice for this batch
                # mode='r+' modifies the existing pre-allocated file
                mmap_ds = open_memmap(outfname_ds, mode='r+')
                mmap_loss = open_memmap(outfname_loss, mode='r+')
                mmap_repair = open_memmap(outfname_repairtime, mode='r+')

                # Write Data
                mmap_ds[start_idx:end_idx, :] = ds_batch
                mmap_loss[start_idx:end_idx, :] = loss_batch
                mmap_repair[start_idx:end_idx, :] = repair_batch

                # FLUSH and CLOSE
                # This ensures the OS does not keep "dirty pages" in RAM
                mmap_ds.flush()
                mmap_loss.flush()
                mmap_repair.flush()
                del mmap_ds, mmap_loss, mmap_repair
                
                # --- MEMORY CLEANUP ---
                del zeta_batch, Xi_batch, eps_batch, lnY_batch, ds_batch, loss_batch, repair_batch
                gc.collect()  # Force garbage collection to free memory immediately
                print_memory(f"{start_idx}, After deletion (After GC)")

            #print(f"Completed Case {i_case}")
            i_case += 1
            print(df_comptime)

    df_comptime.to_csv("./out/trad_computation_time.csv")
    #plt.show()





