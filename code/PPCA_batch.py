# import os
# os.environ["OMP_NUM_THREADS"] = "1"          # OpenMP
# os.environ["OPENBLAS_NUM_THREADS"] = "1"     # OpenBLAS
# os.environ["MKL_NUM_THREADS"] = "1"          # Intel MKL
# os.environ["BLIS_NUM_THREADS"] = "1"         # BLIS
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # Apple Accelerate
# os.environ["NUMEXPR_NUM_THREADS"] = "1"      # numexpr (if you use it)

import numpy as np
import itertools
import pandas as pd
import time
from numpy.lib.format import open_memmap
import os
import gc
from mapping_df import match_fragility, match_loss_ratio, match_repair_time
from GMcorrelation import coord_to_corr_mat_loth2013
from Rrup import calc_Rrup_numba
from gmm import GMM_mean_over_Nobs, GMM_stdev_over_Nobs
from ds_ppca_determ import construct_AAT_for_one_Nobs, step02_calc_c_sigma, step03_construct_Wt, step04_sampling, step05_calc_g_sim_ds
from loss import ds_to_repairtime, ds_to_loss

import psutil
import os

def print_memory(label=""):
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss # Resident Set Size
    mem_mb = mem_bytes / (1024 ** 2)      # Convert to MB
    print(f"[MEMORY] {label}: {mem_mb:.2f} MB")


## Matrix Dimensions ##
# lossratio_ds = (Nb, Nds)
# repairtime_ds = (Nb, Nds)
# cost = (Nb, )
# theta = (Nb, Nds)
# beta = (Nb, Nds)
# asset_coord = (Nb, 3) # 3: Lat, Lon, el
# C = (Nb, Nb)
# L = (Nb, Nb)
# rho_max = scalar
# Mag_list = (Nobs, )
# rup_coord_list = (Nobs, 4, 3) #4: four vertices of rupture, 3: la, lo, el (km)
# Rrup = (Nobs, Nb)
# mu, tau, phi = (Nobs, Nb)
# AAt = (Nb, Nb)
# c_sigma = scalar
# Wt = (Nb, tmax)
# X = (Nobs, tmax)
# noise = (Nobs, Nb)
# ds = (Nobs, Nb)
# loss = (Nobs, Nb)
# repairtime = (Nobs, Nb)

def PPCA_batch(df_fname, Nb_list, Nobs_list, tmax_list,
               magnitude = 7.2,
               rup_coord = np.array([[37.7776, -122.5558, 0],
                                    [37.3604, -122.3447, 0],
                                    [37.3604, -122.3447, 10],
                                    [37.7776, -122.5558, 10]]),
               outdir = "./out/",
               match_frag_fname = "./data/fragility_PGA.csv",
               match_loss_fname = "./data/consequence_repair_PGA.csv",
               match_repair_fname = "./data/consequence_repair_PGA.csv",
               BATCH_SIZE_NOBS=1000,
               ):
    
    outdir_npy = outdir + "/npy/"
    outdir_df = outdir + "/selected_buildings/"

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_npy, exist_ok=True)
    os.makedirs(outdir_df, exist_ok=True)

    
    
    ## Set computation time df##
    Nb_Nobs_combination = list(itertools.product(Nb_list, Nobs_list, tmax_list))
    df_comptime = pd.DataFrame(Nb_Nobs_combination, columns = ["Nb", "Nobs", "tmax"])
    #--------------------------#

    ## Lookup Table ##
    df_lookup = pd.DataFrame(Nb_Nobs_combination, columns = ["Nb", "Nobs", "tmax"])
    df_lookup['i_case'] = range(len(Nb_Nobs_combination))
    df_lookup.to_csv(f'{outdir}/PPCA_lookup.csv', index=False)

    

    
    i_case = 0

    for Nb in Nb_list:

        ## Read building inventory and select first Nb rows ##
        id_sel = range(0, Nb)
        df = pd.read_csv(df_fname)
        df_sel = df[df["id"].isin(id_sel)].copy()
        # ----------------------------#
        
        ## Match the fragility function ##
        df_sel, theta, beta = match_fragility(df_sel, mapping_fname=match_frag_fname)
        #Nds = theta.shape[1]       
        # --------------------------#

        ## Match loss ratio ##
        df_sel, lossratio_ds = match_loss_ratio(df_sel, mapping_fname=match_loss_fname)
        # ---------------------------------#

        ## Match repair time ##
        df_sel, repairtime_ds = match_repair_time(df_sel, mapping_fname=match_repair_fname)
        # --------------------------#
        
        ## Get cost of buildings ##
        cost = df_sel['ReplacementCost'].to_numpy()
        # ----------------#

        ## Write the read df ##
        #f_outname = f"{outdir}/{df_outname}"
        #os.makedirs(outdir_df, exist_ok=True)
        df_outname = "PPCA_" + (df_fname.split('/')[-1]).split('.')[0] +  f"_{i_case}" + ".csv"
        df_sel.to_csv(f"{outdir_df}/{df_outname}", index=False)
        Nb = len(df_sel)
        # ------------------#

        ## Set asset coordinate ##
        asset_coord = np.c_[df_sel['Latitude'], df_sel['Longitude'], np.full(Nb, 0.0)]
        # ------------------------#

        ## Get GM correlation matrix and its cholesky ##
        t = time.time()
        C, L, rho_max = coord_to_corr_mat_loth2013(asset_coord, cholesky=False)
        t_corr_init = time.time() - t 
        # Note: t_corr is calculated once per Nb, we assign it later

        # Run over Nobs list #
        for Nobs in Nobs_list:
            # 1. INITIALIZE ALL MEMMAPS FOR THIS NOBS
            # Create the files on disk once so we can write to them in batches
            active_maps = {}
            # We track time per tmax by using a dictionary
            stats = {tm: {'t_Wt':0, 't_smpl':0, 't_g_ds':0, 't_loss':0, 't_repair':0} for tm in tmax_list}
            sum_t_Rrup = 0; sum_t_mu = 0; sum_t_stdev = 0; sum_t_AAt = 0

            for t_idx, tmax in enumerate(tmax_list):
                case_id = i_case + t_idx
                outfnames = {
                    'ds': f"./out/npy/PPCA_ds_{case_id}_.npy",
                    'loss': f"./out/npy/PPCA_loss_{case_id}_.npy",
                    'repair': f"./out/npy/PPCA_repair_{case_id}_.npy"
                }

                active_maps = {
                    'ds': open_memmap(outfnames['ds'], mode='w+', dtype='int8', shape=(Nobs, Nb)),
                    'loss': open_memmap(outfnames['loss'], mode='w+', dtype='float32', shape=(Nobs, Nb)),
                    #'repair': open_memmap(outfnames['repair'], mode='w+', dtype='float32', shape=(Nobs, Nb))
                }
                del active_maps

                ## BATCH PROCESSING ##
                row_start_idx = 0
                num_batches = int(np.ceil(Nobs / BATCH_SIZE_NOBS))
                
                for b_idx in range(num_batches):
                    # Determine current batch size (last batch might be smaller)
                    curr_Nobs = min(BATCH_SIZE_NOBS, Nobs - row_start_idx)
                    row_end_idx = row_start_idx + curr_Nobs
                    
                    # --------------------------------------#
                    if t_idx ==0:
                        ## Set rupture scenario (For current batch) ##
                        Mag_list = np.full(curr_Nobs, magnitude)
                        rup_coord_list = np.tile(rup_coord, (curr_Nobs, 1, 1)) 
                        # --------------------------------------#

                        ## Calculate Rrup ##
                        if b_idx == 0: calc_Rrup_numba(rup_coord_list[:2], asset_coord[:2]) # Compile once
                        t = time.time()
                        Rrup_list = calc_Rrup_numba(rup_coord_list, asset_coord) 
                        Rrup_list = np.array(Rrup_list)
                        sum_t_Rrup += (time.time() - t)
                        # -------------------------#

                        ## Calculate mean GMI ##
                        t = time.time()
                        mu = GMM_mean_over_Nobs(Mag_list, Rrup_list) 
                        sum_t_mu += (time.time() - t)
                        # -------------------------#

                        ## Calculate stdev GMI ##
                        if b_idx == 0: GMM_stdev_over_Nobs(Mag_list[:2]) # Compile once
                        t = time.time()
                        tau, phi = GMM_stdev_over_Nobs(Mag_list)
                        sum_t_stdev += (time.time() - t)
                        # ---------------------#

                        ## Construct AAT ##
                        tau_mat = np.tile(tau.reshape(curr_Nobs, 1), (1, Nb))
                        phi_mat = np.tile(phi.reshape(curr_Nobs, 1), (1, Nb))
                        #print(tau_mat.nbytes / (1024*1024))
                        #print(phi_mat.nbytes / (1024*1024))

                        t = time.time()
                        # Construct_AAT... handles the current batch size via tau/phi dimensions
                        AAt = construct_AAT_for_one_Nobs(tau_mat[0,:], phi_mat[0,:], C, beta)
                        sum_t_AAt += (time.time() - t)
                        # ----------------------#

                    
                    
                    # ## Loop over tmax (for current batch)
                    # for t_idx, tmax in enumerate(tmax_list):
                    tmax = min(tmax, Nb)
                    # Determine the specific case ID for this tmax configuration
                    # We calculate it dynamically so we can name files correctly on the first pass
                    current_case_id = i_case + t_idx

                    ## Construct Wt ##
                    t = time.time()
                    c_sigma = step02_calc_c_sigma(rho_max, phi_mat, beta)
                    #if tmax > Nb:
                    #    tmax = Nb
                    Wt = step03_construct_Wt(AAt, c_sigma, tmax=tmax)
                    stats[tmax]['t_Wt'] += (time.time() - t)
                    
                    # Store constant c_sigma
                    current_c_sigma = c_sigma 
                    # -------------------#
                    
                    ## Sampling ##
                    t = time.time()
                    X, noise = step04_sampling(curr_Nobs, Nb, tmax)
                    stats[tmax]['t_smpl'] += (time.time() - t)
                    # ----------------#

                    ## Simulate limistate function and damage state ##
                    t = time.time()
                    ds = step05_calc_g_sim_ds(Wt, X, c_sigma, noise, mu, beta, theta)
                    stats[tmax]['t_g_ds'] += (time.time() - t)
                    del X, noise, Wt
                    # -----------------------------------#
                    
                    ## Simulate loss ##
                    t = time.time()
                    loss = ds_to_loss(ds, lossratio_ds, cost)
                    stats[tmax]['t_loss'] += (time.time() - t)
                    # -----------------------#

                    # ## Simulate repair time ##
                    # t = time.time()
                    # repairtime = ds_to_repairtime(ds, repairtime_ds)
                    # stats[tmax]['t_repair'] += (time.time() - t)
                    # # ------------------------#
                    
                    # --- WRITE TO DISK ---
                    m_ds = open_memmap(outfnames['ds'], mode='r+')
                    m_ds[row_start_idx:row_end_idx, :] = ds
                    del ds, m_ds

                    m_loss = open_memmap(outfnames['loss'], mode='r+')
                    m_loss[row_start_idx:row_end_idx, :] = loss
                    del loss, m_loss

                    # m_repair = open_memmap(outfnames['repair'], mode='r+')
                    # m_repair[row_start_idx:row_end_idx, :] = repairtime
                    # del repairtime, m_repair                    
                    
                    gc.collect()
                    

                    row_start_idx += curr_Nobs

                    print_memory(f"Nb:{Nb} Batch:{b_idx+1}/{num_batches}")
                


            # --- CLEANUP & RECORD STATS ---
            for t_idx, tmax_val in enumerate(tmax_list):
                cid = i_case + t_idx
                # Flush and close memmaps

                # Update dataframes
                df_comptime.loc[cid, ['t_corr', 't_Rrup', 't_mu', 't_stdev', 't_AAt']] = [t_corr_init, sum_t_Rrup, sum_t_mu, sum_t_stdev, sum_t_AAt]
                df_comptime.loc[cid, 't_Wt'] = stats[tmax_val]['t_Wt']
                df_comptime.loc[cid, 't_smpl'] = stats[tmax_val]['t_smpl']
                df_comptime.loc[cid, 't_g_ds'] = stats[tmax_val]['t_g_ds']
                df_comptime.loc[cid, 't_loss'] = stats[tmax_val]['t_loss']
                df_comptime.loc[cid, 't_repair'] = stats[tmax_val]['t_repair']
                
                # Save AAt once per case
                np.save(f"./out/npy/PPCA_AAt_{cid}_.npy", AAt)
            #print_memory(f"Nb:{Nb} Batch:{b_idx+1}/{num_batches}")

            i_case += len(tmax_list)
            
    
    df_comptime.to_csv(f"{outdir}/PPCA_computation_time.csv", index=False)








def calc_c_sigma(df_fname, Nb,
               magnitude = 7.2,
               rup_coord = np.array([[37.7776, -122.5558, 0],
                                    [37.3604, -122.3447, 0],
                                    [37.3604, -122.3447, 10],
                                    [37.7776, -122.5558, 10]]),
               outdir = "./out/",
               match_frag_fname = "./data/fragility_PGA.csv"
               ):
    
    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)



    ## Read building inventory and select first Nb rows ##
    id_sel = range(0, Nb)
    df = pd.read_csv(df_fname)
    df_sel = df[df["id"].isin(id_sel)].copy()
    # ----------------------------#
    
    ## Match the fragility function ##
    df_sel, theta, beta = match_fragility(df_sel, mapping_fname=match_frag_fname)
    #Nds = theta.shape[1]       
    # --------------------------#

    ## Set asset coordinate ##
    asset_coord = np.c_[df_sel['Latitude'], df_sel['Longitude'], np.full(Nb, 0.0)]
    # ------------------------#

    ## Get GM correlation matrix and its cholesky ##
    #t = time.time()
    C, L, rho_max = coord_to_corr_mat_loth2013(asset_coord, cholesky=False)
    #t_corr_init = time.time() - t 
    # Note: t_corr is calculated once per Nb, we assign it later
        
    curr_Nobs=2
    ## Set rupture scenario (For current batch) ##
    Mag_list = np.full(curr_Nobs, magnitude)
    rup_coord_list = np.tile(rup_coord, (curr_Nobs, 1, 1)) 
    # --------------------------------------#

    ## Calculate Rrup ##
    #t = time.time()
    Rrup_list = calc_Rrup_numba(rup_coord_list, asset_coord) 
    Rrup_list = np.array(Rrup_list)
    #sum_t_Rrup += (time.time() - t)
    # -------------------------#

    ## Calculate mean GMI ##
    #t = time.time()
    #mu = GMM_mean_over_Nobs(Mag_list, Rrup_list) 
    #sum_t_mu += (time.time() - t)
    # -------------------------#

    ## Calculate stdev GMI ##
    #t = time.time()
    tau, phi = GMM_stdev_over_Nobs(Mag_list)
    #sum_t_stdev += (time.time() - t)
    # ---------------------#

    ## Construct AAT ##
    tau_mat = np.tile(tau.reshape(curr_Nobs, 1), (1, Nb))
    phi_mat = np.tile(phi.reshape(curr_Nobs, 1), (1, Nb))
    #print(tau_mat.nbytes / (1024*1024))
    #print(phi_mat.nbytes / (1024*1024))

    c_sigma = step02_calc_c_sigma(rho_max, phi_mat, beta)
    print(Nb, c_sigma)

    return c_sigma
