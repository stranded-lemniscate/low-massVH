import awkward as ak
from pathlib import Path
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import json

def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

def load_data(file_path):
    return ak.from_parquet(file_path) #, row_groups=15

def load_mc_samples(processes, base_path):
    MC_dict = {}
    for process in processes:
        MC_dict[f'{process}_arr'] = load_data(f'{base_path}/vh_M-{process}_postEE/nominal')
    return MC_dict

def create_plots_dir(plots_dir):
    Path(plots_dir).mkdir(exist_ok=True)

def compute_binning(n_bins, x_low, x_high):
    binning = np.linspace(x_low, x_high, n_bins + 1)
    width = binning[1] - binning[0]
    center = (binning[:-1] + binning[1:]) / 2
    return binning, width, center

def blind_data(data, blind_range):
    return np.where((data > blind_range[0]) & (data < blind_range[1]), np.nan, data)

def compute_histogram(data, binning, weights=None):
    return np.histogram(data, bins=binning, weights=weights)

def extract_variable(arr, var_name):
    if var_name == "mass":
        return np.asarray(arr.mass)
    else:
        parts = var_name.split('/')
        var = np.asarray(getattr(arr, parts[0]))
        for part in parts[1:]:
            var /= np.asarray(getattr(arr, part))
        return var

def sanitize_filename(var_name):
    return var_name.replace('/', '-over-')

def plot_variable(mc_hist, mc_edges, config, var_config, plots_dir): #data_hist, data_edges, 
    plt.style.use(hep.style.CMS)
    # hep.histplot((data_hist, data_edges), histtype='errorbar', yerr=np.sqrt(data_hist), label="Data", color="black")
    # if np.sum(data_hist) == np.sum(mc_hist):
        # mc_label = f"MC: Normalised to data"
    # else:
    mc_label = f"MC: Norm of {var_config['mc_norm']} events"

    hep.histplot((mc_hist, mc_edges), histtype='step', label=mc_label)
    plt.xlabel(var_config['xlabel'])
    plt.ylabel(var_config['ylabel'])
    if config['plot_settings']['legend']:
        plt.legend()
    plt.xlim(var_config['plot_range'])
    plt.ylim(bottom=config['plot_settings']['ylim_bottom'])
    hep.cms.label(config['plot_settings']['cms_label'], data=True, year=config['plot_settings']['cms_year'], com=config['plot_settings']['cms_com'])
    plt.tight_layout()
    sanitized_var_name = sanitize_filename(var_config['name'])
    plt.savefig(f"{plots_dir}/{sanitized_var_name}_plot.pdf")
    return plt.show()
    plt.clf()