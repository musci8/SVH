import os
import numpy as np
from scipy.special import comb
import pandas as pd

methods = ["svmis", "svh"]
Ns = [100, 200, 500]  # number of nodes
densities = [0.005, 0.01, 0.02]  # density values
closures = [0.0, 0.25, 0.5, 0.75, 1.0]  # closure probabilities
fs = np.linspace(0.4, 1, 7)
replicates = 100  # number of independent runs per configuration; increase for more precise estimates
max_size_implanted_sets = 4
max_size = 6  # maximum number of additional nodes attached to the core group
n_interactions = 20
out_table_file = "benchmark_params_table.json"

def main():

    if os.path.exists(out_table_file):
        print(f"Output file {out_table_file} already exists. Please remove it before running the script.")
        return

    params_table_df = []
    for method in methods:
        for N in Ns:
            for density in densities:
                T = int(density * comb(N, 2))
                for closure in closures:
                    for f in fs:
                        param_dict = {
                            "method": method,
                            "N": N,
                            "density": density,
                            "closure": closure,
                            "f": f,
                            "T": T,
                            "max_size_implanted_sets": max_size_implanted_sets,
                            "max_size": max_size,
                            "n_interactions": n_interactions,
                            "n_replicates": replicates
                        }
                        params_table_df.append(param_dict)

    params_table_df = pd.DataFrame(params_table_df)

    # save table
    params_table_df.to_json(out_table_file, orient='records', lines=True)

if __name__ == "__main__":
    main()