import numpy as np
import pandas as pd

def match_fragility(df_sel, mapping_fname = "./out_R2D/fragility_PGA.csv"):
    HClass2abb = {
                        'Low-Rise':'L',
                        'Mid-Rise':'M',
                        'High-Rise':'H',
                        np.nan :'',
                    }
    DL2abb = {
                'Pre-Code': 'PC',
                'Low-Code': 'LC',
                'Moderate-Code': 'MC',
                'High-Code': 'HC',
                np.nan: '',
            }
    id_frag = "LF." + df_sel['StructureType'] + "." + df_sel['HeightClass'].map(HClass2abb) + "." + df_sel['DesignLevel'].map(DL2abb)
    id_frag = id_frag.str.replace('..', '.', regex=False)
    df_sel = df_sel.copy()
    df_sel['ID_fragility'] = id_frag
    
    df_frag_mapping = pd.read_csv(mapping_fname)
    #col_theta = ["LS1-Theta_0", "LS2-Theta_0", "LS3-Theta_0", "LS4-Theta_0"]
    #col_beta = ["LS1-Theta_1", "LS2-Theta_1", "LS3-Theta_1", "LS4-Theta_1"]
    
    df_sel['theta1'] = df_sel['ID_fragility'].map(df_frag_mapping.set_index('ID')['LS1-Theta_0'])
    df_sel['beta1'] = df_sel['ID_fragility'].map(df_frag_mapping.set_index('ID')['LS1-Theta_1'])
    df_sel['theta2'] = df_sel['ID_fragility'].map(df_frag_mapping.set_index('ID')['LS2-Theta_0'])
    df_sel['beta2'] = df_sel['ID_fragility'].map(df_frag_mapping.set_index('ID')['LS2-Theta_1'])
    df_sel['theta3'] = df_sel['ID_fragility'].map(df_frag_mapping.set_index('ID')['LS3-Theta_0'])
    df_sel['beta3'] = df_sel['ID_fragility'].map(df_frag_mapping.set_index('ID')['LS3-Theta_1'])
    df_sel['theta4'] = df_sel['ID_fragility'].map(df_frag_mapping.set_index('ID')['LS4-Theta_0'])
    df_sel['beta4'] = df_sel['ID_fragility'].map(df_frag_mapping.set_index('ID')['LS4-Theta_1'])

    theta = np.c_[df_sel['theta1'].to_numpy().astype(float),
                        df_sel['theta2'].to_numpy().astype(float),
                        df_sel['theta3'].to_numpy().astype(float),
                        df_sel['theta4'].to_numpy().astype(float)]
    beta =  np.c_[df_sel['beta1'].to_numpy().astype(float),
                    df_sel['beta2'].to_numpy().astype(float),
                    df_sel['beta3'].to_numpy().astype(float),
                    df_sel['beta4'].to_numpy().astype(float)]  

    return df_sel, theta, beta

def match_loss_ratio(df_sel, mapping_fname = "./out_R2D/consequence_repair_PGA.csv"):
    ## Match loss ratio##
    df_frag_mapping = pd.read_csv(mapping_fname)
    #col_loss = ["DS1-Theta_0","DS2-Theta_0","DS3-Theta_0","DS4-Theta_0"]

    id_lossratio = "LF." + df_sel['OccupancyClass'] + "-Cost"
    id_lossratio = id_lossratio.str.replace('..', '.', regex=False)
    df_sel = df_sel.copy()
    df_sel['ID_lossratio'] = id_lossratio
    
    df_sel['DS1_lossratio'] = df_sel['ID_lossratio'].map(df_frag_mapping.set_index('ID')['DS1-Theta_0'])
    df_sel['DS2_lossratio'] = df_sel['ID_lossratio'].map(df_frag_mapping.set_index('ID')['DS2-Theta_0'])
    df_sel['DS3_lossratio'] = df_sel['ID_lossratio'].map(df_frag_mapping.set_index('ID')['DS3-Theta_0'])
    df_sel['DS4_lossratio'] = df_sel['ID_lossratio'].map(df_frag_mapping.set_index('ID')['DS4-Theta_0'])

    lossratio = np.c_[df_sel['DS1_lossratio'].to_numpy().astype(float),
                    df_sel['DS2_lossratio'].to_numpy().astype(float),
                    df_sel['DS3_lossratio'].to_numpy().astype(float),
                    df_sel['DS4_lossratio'].to_numpy().astype(float)]
    return df_sel, lossratio

def match_repair_time(df_sel, mapping_fname = "./out_R2D/consequence_repair_PGA.csv"):
    df_frag_mapping = pd.read_csv(mapping_fname)
    #col_loss = ["DS1-Theta_0","DS2-Theta_0","DS3-Theta_0","DS4-Theta_0"]

    id_lossday = "LF." + df_sel['OccupancyClass'] + "-Time"
    id_lossday = id_lossday.str.replace('..', '.', regex=False)
    df_sel = df_sel.copy()
    df_sel['ID_lossday'] = id_lossday

    df_sel['DS1_lossday'] = df_sel['ID_lossday'].map(df_frag_mapping.set_index('ID')['DS1-Theta_0'])
    df_sel['DS2_lossday'] = df_sel['ID_lossday'].map(df_frag_mapping.set_index('ID')['DS2-Theta_0'])
    df_sel['DS3_lossday'] = df_sel['ID_lossday'].map(df_frag_mapping.set_index('ID')['DS3-Theta_0'])
    df_sel['DS4_lossday'] = df_sel['ID_lossday'].map(df_frag_mapping.set_index('ID')['DS4-Theta_0'])

    lossday = np.c_[df_sel['DS1_lossday'].to_numpy().astype(float),
                    df_sel['DS2_lossday'].to_numpy().astype(float),
                    df_sel['DS3_lossday'].to_numpy().astype(float),
                    df_sel['DS4_lossday'].to_numpy().astype(float)]
    
    return df_sel, lossday