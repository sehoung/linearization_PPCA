import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def data_to_cdf(x):
    x_sorted = np.sort(x)
    cdf = np.arange(1, len(x_sorted)+1) / len(x_sorted)
    return x_sorted, cdf

params = {
    'font.size': 9,          # General default font size
    'axes.titlesize': 9,     # Subplot titles (e.g. ax.set_title)
    'axes.labelsize': 9,     # Axis labels (e.g. ax.set_xlabel)
    'xtick.labelsize': 9,    # X-axis tick labels
    'ytick.labelsize': 9,    # Y-axis tick labels
    'legend.fontsize': 9,    # Legend text
    'figure.titlesize': 9,   # Figure main title (e.g. fig.suptitle)
    'font.family': 'sans-serif' # Optional: Set font family
}
plt.rcParams.update(params)


os.makedirs("./out/figdata", exist_ok=True)


#### SF Downtown ####
trad_lookup = pd.read_csv(f"./out/trad_lookup.csv")
icase_trad_sel = trad_lookup['i_case'].to_numpy()

for icase in icase_trad_sel:
    loss = np.load(f"./out/npy/trad_loss_{icase}_.npy")
    
    loss_tot_trad = np.sum(loss, axis=1)
    x, cdf_trad = data_to_cdf(loss_tot_trad)
    pe_trad = 1-cdf_trad
    
    np.savetxt(f"./out/figdata/trad_loss_ratio_sf_{icase}.txt", np.c_[x, pe_trad])
print("trad Done!")


PPCA_lookup = pd.read_csv(f"./out/PPCA_lookup.csv")
icase_PPCA_sel = PPCA_lookup['i_case'].to_numpy()
for i, icase in enumerate(icase_PPCA_sel):
    loss = np.load(f"./out/npy/PPCA_loss_{icase}_.npy")
    print(icase, np.shape(loss))
    print(icase, np.sum(np.all(loss == 0, axis=1)))

    loss_tot_PPCA = np.sum(loss, axis=1)
    
    x, cdf_PPCA = data_to_cdf(loss_tot_PPCA)
    pe_PPCA = 1-cdf_PPCA

    np.savetxt(f"./out/figdata/PPCA_loss_ratio_sf_{icase}.txt", np.c_[x, pe_trad])
print("PPCA Done!")


