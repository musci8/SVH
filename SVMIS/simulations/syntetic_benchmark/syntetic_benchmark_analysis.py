import os
import json
import argparse
import numpy as np
from scipy import special
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hypergraphx as hgx

from tqdm.notebook import tqdm

import sys
sys.path += ["../../"]

from src.synthetic_bentchmark import create_benchmark
from src.filters import get_svh, get_svmis, fdr_correction

ALPHAS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]


def read_parameters(params_file, idx):

    with open(params_file, 'r') as f:
        for i, line in enumerate(f):
            if i == idx:
                param_dict = json.loads(line)
                return param_dict


def true_positive_rate(true,pred):
    return len(set(pred).intersection(true))/len(true) if len(true) > 0 else 0.0


def precision(true, pred):
    return len(set(pred).intersection(true))/len(pred) if len(pred) > 0 else 0.0


def false_discovery_rate(true,pred):
    return len(set(pred).difference(true))/len(pred) if len(pred) > 0 else 0.0


def jaccard_index(true,pred):
    return len(set(true).intersection(pred))/len(set(true).union(pred))


def evaluate_performance(true_groups, pred_groups):

    tpr = true_positive_rate(true_groups, pred_groups)
    fdr = false_discovery_rate(true_groups, pred_groups)
    prec = precision(true_groups, pred_groups)
    jacc = jaccard_index(true_groups, pred_groups)

    line = {"tpr": tpr, "fdr": fdr, "precision": prec, "jaccard": jacc}

    return line


def evaluate_svh(svh_dict, true_groups):

    # get number of nodes involved for each size
    n_nodes_involved_size = {size: df_res["group"].explode().nunique() for size, df_res in svh_dict.items()}

    # get validated groups for each alpha
    results_svh = []
    for alpha in ALPHAS:
        pred_svh = []
        for size, df_res in svh_dict.items():
            n_nodes_involved = n_nodes_involved_size[size]
            n_tests = special.binom(n_nodes_involved, size)
            df_res.loc[:, "fdr"] = fdr_correction(df_res["pvalue"], alpha=alpha, n_tests=n_tests)

            pred_svh.extend([tuple(sorted(g)) for g in df_res.query("fdr == True")['group']])

        res_dict_svh_alpha = evaluate_performance(true_groups, pred_svh)
        res_dict_svh_alpha = {"alpha": alpha, **res_dict_svh_alpha}

        results_svh.append(res_dict_svh_alpha)

    return results_svh


def evaluate_svmis(svmis_dict, true_groups):

    # get total number of nodes
    n_nodes = len(set(node for size, df_res in svmis_dict.items() for node in df_res["group"].explode().unique()))

    # get validated groups for each alpha
    results_svmis = []
    for alpha in ALPHAS:
        pred_svmis = []
        for size, df_res in svmis_dict.items():
            n_tests = special.binom(n_nodes, size)
            df_res.loc[:, "fdr"] = fdr_correction(df_res["pvalue"], alpha=alpha, n_tests=n_tests)

            pred_svmis.extend([tuple(sorted(g)) for g in df_res.query("fdr == True")['group']])

        res_dict_svmis = evaluate_performance(true_groups, pred_svmis)
        res_dict_svmis = {"alpha": alpha, **res_dict_svmis}

        results_svmis.append(res_dict_svmis)

    return results_svmis


def from_pandas_to_hypergraph(df):
    """ Create a hypergraphx Hypergraph from a pandas DataFrame with columns 'a' and 'b', where 'a' contains the nodes and 'b' contains the groups. Each group in 'b' corresponds to a hyperedge connecting the nodes in 'a'. The weight of each hyperedge is the number of times that group appears in the DataFrame.
    """

    H = hgx.Hypergraph(weighted=True)
    for group, rows in df.groupby("b"):
        group = sorted(rows.a.tolist())
        if not H.check_edge(group):
            H.add_edge(group, weight=1)
        else:
            w = H.get_weight(group)
            H.set_weight(group, w+1)

    return H
    


def main(args):
    
    # read parameters and fix seed
    params_dict = read_parameters(args.params_file, args.experiment_n)

    n_replicates = params_dict.pop("n_replicates")
    for iter in range(n_replicates):

        # set random seed
        seed = args.experiment_n * 10_000 + iter
        RNG = np.random.RandomState(seed)

        # generate synthetic benchmark
        df_hyper, implanted_groups = create_benchmark(N=params_dict["N"], T=params_dict["T"], max_size_implanted_sets=params_dict["max_size_implanted_sets"], closure=params_dict["closure"], max_size=params_dict["max_size"], f=params_dict["f"], n_interactions=params_dict["n_interactions"], rng=RNG)
        
        true_groups = [tuple(sorted(g)) for g in implanted_groups]

        # to hgx hypergraph
        H = from_pandas_to_hypergraph(df_hyper)

        results = []

<<<<<<< Updated upstream
        # svh
        svh_dict = get_svh(H, alpha=0.01, max_size=4, verbose=False)
        res_svh = evaluate_svh(svh_dict, true_groups)
        for res in res_svh:
            results.append({"method": "svh", **params_dict, "iter":iter, "seed":seed, **res})

        # statistically validated sets (approximate)
        svmis_dict = get_svmis(H, alpha=0.01, min_size=2, max_size=4, approximate=True, verbose=False)
        res_svmis = evaluate_svmis(svmis_dict, true_groups)
        for res in res_svmis:
            results.append({"method": "svmis", **params_dict, "iter":iter, "seed":seed, **res})
=======
        if params_dict["method"] == "svh":
            # svh
            svh_dict = get_svh(H, alpha=0.01, max_size=4, verbose=False)
            res_svh = evaluate_svh(svh_dict, true_groups)
            for res in res_svh:
                results.append({"method": "svh", **params_dict, "iter":iter, **res})

        elif params_dict["method"] == "svmis":
            # statistically validated sets (approximate)
            svmis_dict = get_svmis(H, alpha=0.01, min_size=2, max_size=4, approximate=True, verbose=False)
            res_svmis = evaluate_svmis(svmis_dict, true_groups)
            for res in res_svmis:
                results.append({"method": "svmis", **params_dict, "iter":iter, **res})

        else:
            raise ValueError(f"Unknown method: {params_dict['method']}")
>>>>>>> Stashed changes

        # store results
        with open(args.output_file, 'a') as ww:
            for res in results:
                line = json.dumps(res)
                ww.write(line + '\n')


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Run synthetic benchmark analysis.')
    parser.add_argument('--params-file', type=str, default='benchmark_params_table.json', help='Path to the parameters JSON file.')
    parser.add_argument('--output-file', type=str, default='benchmark_analysis_output', help='File to save analysis results.')
    parser.add_argument('--experiment-n', type=int, default=0, help='Index of the experiment to analyze (corresponding to a line in the parameters file).')
    args = parser.parse_args()

    # if results directory does not exist, create it
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    main(args)