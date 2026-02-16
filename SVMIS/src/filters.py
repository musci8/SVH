from collections import OrderedDict, Counter, defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats, special
from tqdm.auto import tqdm

def _get_bipartite_representation(hypergraph):
    """ 
    Transform the hypergraph into a DataFrame representing its bipartite representation. Column "a" indicate nodes and column "b" indicates the hyperedges.
    """
    
    edge_index = 0
    bipartite_list = []
    for edge in hypergraph.get_edges():

        w = hypergraph.get_weight(edge)
        for _ in range(w):
            for node in edge:
                bipartite_list.append((node, edge_index))
            edge_index += 1

    bipartite_df = pd.DataFrame(bipartite_list, columns=["a", "b"])

    return bipartite_df


def intersection_distribution(set_sizes, N):
    """
    Compute full probability distribution of intersection size for k sets using iterative convolution of hypergeom distributions.

    Parameters
    ----------
    set_sizes : list of ints
        Sizes of each set: [n1, n2, ..., nk]
    N : int
        The total number of groups in the hypergraph.

    Returns
    -------
    probs : 1D numpy array
        Probability p(x) for x = 0 ... min(set_sizes)
    """
    # distribution for the first two sets
    n1, n2 = set_sizes[0], set_sizes[1]

    max12 = min(n1, n2)
    p12 = np.zeros(max12 + 1)

    # hypergeometric distribution for intersection of first two sets
    for x in range(max12 + 1):
        p12[x] = stats.hypergeom.pmf(x, N, n1, n2)

    current_dist = p12

    # iteratively convolve with each additional set
    for i in range(2, len(set_sizes)):
        n_i = set_sizes[i]

        # new distribution size <= previous intersection potential
        max_prev = len(current_dist) - 1
        max_new = min(max_prev, n_i)

        new_dist = np.zeros(max_new + 1)

        # recursion:
        # p_{1..i}(y) = sum_x p_{1..i-1}(x) * H(y | N, x, n_i)
        for y in range(max_new + 1):
            total = 0.0
            for x in range(y, max_prev + 1):
                total += current_dist[x] * stats.hypergeom.pmf(y, N, x, n_i)
            new_dist[y] = total

        current_dist = new_dist

    return current_dist


def _pvalue_svmis_exact(group, node_neighbors, n_groups):
    """
    Compute the p-value of the observed intersection
    of a list of neighbor lists, using iterative hypergeometric convolution.

    Parameters
    ----------
    group : tuple of ints
        Nodes in the group whose intersection is tested.
    node_neighbors : dict
        Mapping node -> list of participating hyperedges.
    n_groups : int
        Total number of hyperedges.

    Returns
    -------
    group_pval : float
        P-value for the group considered: P(X >= observed intersection)
    """

    # get the hyperedges where each node in the group participates and get their sizes
    neigh_lists = [node_neighbors[node] for node in group]
    set_sizes = [len(s) for s in neigh_lists]

    # compute how many times all nodes in the group are observed together
    intersection = set(neigh_lists[0])
    for s in neigh_lists[1:]:
        intersection &= set(s)
    observed = len(intersection)

    # compute probability that the number of hyperedges containing all nodes in the group is X and corresponding p-value
    dist = intersection_distribution(set_sizes, n_groups)
    group_pval = np.sum(dist[observed:])

    return group_pval


def _pvalue_svmis_approx(t):
    """
    Return approximated p-value.
    
    Parameters
    ----------
    t   :tuple
        Tuple containing (N12, N, N1, N2, ... Nn)
        
    Returns
    -------
    p-value:    float
                approximated p-value associated to the input tuple
    """
    n12 = t[0] # how many times the current group is observed together
    n = t[1] # total number of hyperlinks
    ns = np.array(t[2:]) # number of times each node in the current group is observed in any hyperlink

    p = np.prod(ns / n)
    p = stats.binom.sf(n12 - 1, p=p, n=n)

    if p >= 0:
        return p
    else:
        return np.clip(p, 0, 1)
    

def fdr_correction(pvalues, n_tests, alpha):

    alpha_bonf = alpha / n_tests

    ps = np.sort(pvalues.values)
    k = np.arange(1, len(ps) + 1) * alpha_bonf

    is_p_smaller = ps < k

    if is_p_smaller.any():
        fdr = k[is_p_smaller][-1] 
    else:
        fdr = 0

    return pvalues < fdr



def get_svmis(H, min_size=2, max_size=0, alpha=0.01, approximate=True):

    """
    Extract the Statistically Validated Maximal Interacting Sets.

    Parameters
    -------------
    H: hypergraphx.Hypergraph
        The input hypergraph

    min_size:	int
        Minimum size of the simplices to be tested

    max_size:	int
        Maximum size of the simplices to be tested

    alpha: float
        Significance level.

    approximate: bool
        Whether to use approximate p-value. At the moment only approximated version is implemented

    n_workers: int


    Returns
    -------------
    svs:		DataFrame
        The DataFrame is a Table with columns ['group','pvalue','fdr']. 
        'group' contains all the simplices (mapped as tuples) tested in the hypergraph
        'pvalue' reports the pvalue
        'fdr' is a bool that is True if the simplex has been validated, False otherwise
    """

    max_hyperedge_size = H.max_size()

    # check max size not larger than largest hyperedge size
    if max_size != 0:
        max_size = min(max_size, max_hyperedge_size)
    else:
        max_size = max_hyperedge_size

    # transform the hypergraph in its bipartite representation
    df = _get_bipartite_representation(H)

    # get all hyperedges considering their multiplicity
    all_hyperedges = df.groupby('b')['a'].apply(lambda x: tuple(sorted(x))).tolist()
    N = len(all_hyperedges)
    num_nodes = df.a.nunique()

    # get the neighbors of each node in a dict
    neigh_set_a_sub = dict(df.groupby('a')['b'].apply(list).reset_index().values)

    significant_sets = []
    svmis = {}
    for size in tqdm(list(range(min_size, max_size+1))[::-1], total=max_size - min_size, leave=False):

        # generate all the subgroups of size 'size' from the already created groups
        # if a set s is deemed significant, we do not test all the subsets of size size of the set s
        drop = set()
        #for l in list(map(lambda x: tuple(combinations(x, size)), significant_sets)):
        #    drop.update(l)
        for s in significant_sets:
            if len(s) >= size:
                for comb in combinations(s, size):
                    #drop.add(tuple(sorted(comb)))
                    drop.add(tuple(comb))

        groups_size = filter(lambda x: len(x) >= size, all_hyperedges)
        
        #p = Pool(processes=cpu_count())
        if not approximate:
        
            groups = set()

            for l in (map(lambda x: tuple(combinations(x, size)), groups_size)): 
                for g in l: 
                    groups.add(g)
            groups = groups.difference(drop)

            # compute p values
            pvalues = dict(zip(groups, 
                               #p.map(_pvalue_intersect,
                               map(_pvalue_svmis_exact,
                                      zip(groups, [neigh_set_a_sub]*len(groups), [N]*len(groups)))))
                    
        else:
            deg_a = Counter(df.a)
            groups = defaultdict(int)

            # for each larger group, generate all possible combinations of size 'size' and count their occurrences
            # for l in map(lambda x: tuple(combinations(x, size)), groups_size): 
            #     for g in l: 
            #         if g not in drop: 
            #             groups[g] += 1
            for edge in groups_size:
                for comb in combinations(edge, size):
                    if comb not in drop:
                        #groups[tuple(sorted(comb))] += 1
                        groups[tuple(comb)] += 1

            pvalues = dict(zip(groups, 
                               #p.map(p_appr,
                                map(_pvalue_svmis_approx,
                                     [tuple([groups[i], N]) + tuple([deg_a[ii] for ii in i]) for i in groups])))

        #p.close()

        temp_df = pd.DataFrame(pvalues.items())
        temp_df.columns = ['group','pvalue']

        # apply fdr correction
        n_tests = special.binom(num_nodes, size)
        temp_df.loc[:, "fdr"] = fdr_correction(temp_df.pvalue, n_tests, alpha=alpha)
        
        svmis[size] = temp_df

        # update the list of significant sets
        significant_sets_size = temp_df.query('fdr').group.tolist()
        significant_sets.extend(significant_sets_size)

    return svmis

