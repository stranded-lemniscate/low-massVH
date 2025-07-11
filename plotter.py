import argparse
import numpy as np
from plot_utils import *

def main(config_file, input_dir, output_dir):
    # Load the configuration settings
    config = load_config(config_file)
    print(f"INFO: Loaded configuration from {config_file}")
    
    # Load the nominal data array
    # data_arr = load_data(f'{input_dir}/DataC_2022/nominal')
    # print(f"INFO: Loaded data from {input_dir}/DataC_2022/nominal")  
    
    # List of processes to consider
    processes = ['70']#['110', '105', '95','90','60'] #'ggh', 'vbf', vh , 'tth'   ,
    
    # Load Monte Carlo samples for each process
    MC_dict = load_mc_samples(processes, input_dir)
    print(f"INFO: Loaded MC samples for processes: {processes}")
    
    # Create directory for output plots if it doesn't exist
    create_plots_dir(output_dir)
    print(f"INFO: Output directory created at {output_dir}")

    # Define cross-sections for each process
    cross_sections = {
        # 'ggh': 52.23,
        # 'vbf': 4.078,
        'vh': 1.457 + 0.9439,
        # 'tth': 0.57
    }

    # Compute MC weights for each process based on cross-sections
    mc_weights = []
    for process in processes:
        weight = np.asarray(MC_dict[f'{process}_arr'].weight) * cross_sections['vh'] #process
        mc_weights.append(weight)
    mc_weights_combined = np.concatenate(mc_weights)
    print("Combined MC weights computed.")

    # Iterate over each variable in the configuration
    for var_config in config['variables']:
        var_name = var_config['name']
        print("------------------------------------------")
        print(f"INFO: Now processing variable: {var_name}")
        
        # Extract the variable from the data array
        # data_var = extract_variable(data_arr, var_name)
        
        # Extract the variable for each process and combine them
        mc_var_combined = [extract_variable(MC_dict[f'{process}_arr'], var_name) for process in processes]

        # Blind the data in the specified range
        # data_var = blind_data(data_var, config['blind_range'])

        # Compute binning
        binning, width, center = compute_binning(var_config['n_bins'], var_config['hist_range'][0], var_config['hist_range'][1])

        # Compute histograms for data and combined MC variables
        # data_hist, data_edges = compute_histogram(data_var, binning)

        mc_var_combined = np.concatenate(mc_var_combined)
        mc_var_hist, mc_var_edges = compute_histogram(mc_var_combined, binning, weights=mc_weights_combined)
        mc_var_hist /= np.sum(mc_var_hist)

        # Normalize MC histogram to data if required
        if var_config['normalise_mc_to_data']:
            # mc_var_hist *= np.sum(data_hist)
            print(f"INFO: MC histogram normalized to data.")
        else:
            mc_var_hist *= var_config['mc_norm']
            print(f"INFO: MC histogram normalized with factor: {var_config['mc_norm']}")

        # Plot the variable
        plot_variable(mc_var_hist, mc_var_edges, config, var_config, output_dir) #data_hist, data_edges, 
        print(f"INFO: Plot saved for variable {var_name}")

if __name__ == "__main__":
    # Set up argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Plotting script")
    parser.add_argument('config_file', type=str, help="Path to the configuration file")
    parser.add_argument('input_dir', type=str, help="Directory containing the input samples")
    parser.add_argument('output_dir', type=str, help="Directory to save the output plots")

    # Parse the command line arguments
    args = parser.parse_args()
    print(f"INFO: Arguments received: config_file={args.config_file}, input_dir={args.input_dir}, output_dir={args.output_dir}")
    
    main(args.config_file, args.input_dir, args.output_dir)