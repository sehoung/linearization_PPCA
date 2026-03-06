import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

os.makedirs("./out/figs", exist_ok=True)


# 1. Load lookups
trad_lookup = pd.read_csv("./out/trad_lookup.csv")
PPCA_lookup = pd.read_csv("./out/PPCA_lookup.csv")

# 2. Identify unique Nb values present in PPCA
unique_nb_ppca = PPCA_lookup['Nb'].unique()

for nb_val in unique_nb_ppca:
    # Get subset of lookup for this Nb
    subset_ppca = PPCA_lookup[PPCA_lookup['Nb'] == nb_val]
    cases_ppca = subset_ppca['i_case'].unique()
    
    # Get Trad cases for the same Nb
    cases_trad = trad_lookup[trad_lookup['Nb'] == nb_val]['i_case'].unique()
    
    m = len(cases_ppca)
    if m == 0: continue
    
    # Grid setup: 3 columns wide
    cols = min(3, m)
    rows = (m + cols - 1) // cols
    
    # Initialize Figure and Axes
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    fig.suptitle(f"Exceedance Curves | Nb: {nb_val}", fontsize=12, fontweight='bold')
    
    axes_flat = axes.flatten()

    for idx, i_ppca in enumerate(cases_ppca):
        ax = axes_flat[idx]
        
        # Get tmax for this specific i_case from the PPCA lookup
        tmax_val = subset_ppca[subset_ppca['i_case'] == i_ppca]['tmax'].iloc[0]
        
        # --- Plot Traditional curves as reference ---
        for i_trad in cases_trad:
            trad_path = f"./out/figdata/trad_loss_ratio_sf_{i_trad}.txt"
            if os.path.exists(trad_path):
                t_data = np.loadtxt(trad_path)
                ax.plot(t_data[:, 0], t_data[:, 1], color='gray', linestyle='-', 
                        alpha=0.5, label=f'Trad Case {i_trad}' if m < 4 else "")

        # --- Plot the PPCA curve ---
        ppca_path = f"./out/figdata/PPCA_loss_ratio_sf_{i_ppca}.txt"
        if os.path.exists(ppca_path):
            p_data = np.loadtxt(ppca_path)
            ax.plot(p_data[:, 0], p_data[:, 1], color='red', linestyle='--', 
                    linewidth=2, label=f'PPCA (i:{i_ppca})')

        # Formatting each axis
        ax.set_yscale('log')
        ax.set_title(f"PPCA, t={tmax_val}")
        ax.set_xlabel("Loss (USD)")
        ax.set_ylabel("Probability of Exceedance")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        if m < 6: 
            ax.legend(fontsize=8, loc='upper right')

    # Hide unused axes
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = f"./out/figs/Comparison_Nb_{nb_val}.png"
    plt.savefig(output_filename, dpi=600) # Uncomment to save
    print(f"Generated figure: {output_filename}")

plt.show()

print("Processing complete.")