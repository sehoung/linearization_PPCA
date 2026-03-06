import numba
import numpy as np

@numba.njit
def ds_to_loss(ds1, lossratio1, cost1):
    # ds = (Nobs, Nb)
    # lossratio = (Nb, Nds)
    # cost = (Nb, )
    Nmc = np.shape(ds1)[0]
    Nb = np.shape(ds1)[1]
    loss = np.empty((Nmc, Nb))
    for i_mc in range(Nmc):
        for i_b in range(Nb):
            ds_ij = ds1[i_mc, i_b]
            if ds_ij ==0:
                lossratio_ij = 0    
            else:
                lossratio_ij = lossratio1[i_b, ds_ij-1]
            loss_ij = lossratio_ij * cost1[i_b]
            loss[i_mc, i_b] = loss_ij
    return loss

@numba.njit
def ds_to_repairtime(ds1, repairtime_b_ds):
    # ds = (Nobs, Nb)
    # lossratio = (Nb, Nds)
    # cost = (Nb, )
    Nmc = np.shape(ds1)[0]
    Nb = np.shape(ds1)[1]
    repairtime = np.empty((Nmc, Nb))
    
    for i_mc in range(Nmc):
        for i_b in range(Nb):
            ds_ij = ds1[i_mc, i_b]
            if ds_ij ==0:
                repairtime_ij = 0    
            else:
                repairtime_ij = repairtime_b_ds[i_b, ds_ij-1]
            
            repairtime[i_mc, i_b] = repairtime_ij
    return repairtime