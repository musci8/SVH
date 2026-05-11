from itertools import combinations
import numpy as np
from scipy import special, stats
import pandas as pd


def create_benchmark(N, T, max_size_implanted_sets, closure, f=None, max_size=10, n_interactions=20, rng=None):
    """
    Create a synthetic hypergraph benchmark by generating implanted sets and then creating hyperedges by adding random nodes to those sets.

    Parameters
    -------------
    N: int,
        Number of nodes in the hypergraph.
    T: int,
        Number of implanted sets to create for each order. Note that this is not the actual number of sets created, as some may be filtered out if they are proper faces of larger sets or some may be generated through closure.
    max_size_implanted_sets: int,
        Maximum size of the implanted sets.
    closure: float,
        The fraction of implanted sets generated through closure.
    f: float or None,
        Fraction of interactions in which an implanted set is expanded into a larger hyperedge by adding n randomly selected nodes, with n drawn from U(0, max_size - len(implanted_set)). If None, implanted sets are expanded by sampling the number of additional nodes n from U(0, max_size - len(implanted_set)) for all interactions.
    max_size: int,
        Maximum size of additional interactions.
    n_interactions: int,
        Number of interactions per group.
    rng: np.random.RandomState, optional
        Random number generator for reproducibility. If None, a default generator with seed 42 will be used.

    Returns
    -------------
    df: pd.DataFrame,
        DataFrame containing the edges of the hypergraph in a node-group bipartite representation.
    groups: list of tuples,
        List of implanted sets (groups) that were generated, after filtering out those that are proper faces of larger groups.
    """

    if rng is None:
        rng = np.random.RandomState(42)
    
    groups = []
    for size in range(2, max_size_implanted_sets + 1):

        # when generating sets through closure, each clique of size n+1 generates n+1 groups
        # thus, the number of such cliques to be generated is equal to T / (n+1)
        n_cliques = int(T / (size + 1))
        
        if n_cliques == 0:
            raise ValueError(f"Too few sets requested for size {size}. Increase T or decrease max_size_implanted_sets.")

        # get the number of implanted sets to be generated via closure
        num_closed = int(closure * n_cliques)
        num_open = n_cliques - num_closed
             
        closed = 0
        all_g = 0
        # create closed simplices (i.e. size+1 sets for each num_closed)
        for _ in range(num_closed):
            clique_nodes = rng.choice(range(N), replace=False, size=size+1)
            clique_nodes = tuple(sorted(clique_nodes))

            # add all the subsets of size n
            for g in combinations(clique_nodes, size):
                groups.append(g)
                closed += 1
                all_g += 1

        # create groups that are not closed (i.e. (size + 1) * num_open)
        for _ in range(num_open):
            for __ in range(size + 1):
                g = rng.choice(range(N), replace=False, size=size)
                groups.append(tuple(sorted(g)))
                all_g += 1
                    
    # eliminates groups that are proper faces of larger groups
    # i.e. if we have (1, 2) and (1, 2, 3) as groups, we remove (1, 2)
    for size in range(2, max_size_implanted_sets + 1):
        larger_g = set()
        all_larger_groups = filter(lambda x: len(x) > size, groups)

        # generate all possible combinations of order 'order' from larger groups
        for l in map(lambda x: tuple(combinations(x, size)), all_larger_groups):
            for g in l: 
                 larger_g.add(g)

        # filter out the smaller groups that are faces of larger ones
        groups = list(filter(lambda x: (x not in larger_g), groups ))

    groups = list(set(groups))  

    # create hyperedges for each of the created groups
    interactions = []
    for g in groups:

        # generate l interactions sampled from binomial
        n_edges_to_create = stats.binom.rvs(p=.5, n=n_interactions, random_state=rng)

        if f is not None:
            # use the parameter f to control how often implanted sets are diluted
            # note that the sampling of the size starts from 1, as we want to ensure that the implanted set is always part of the interaction

            # get the fraction of those to be expanded into larger hyperedges
            n_same = int((1 - f) * n_edges_to_create)
            n_to_expand = n_edges_to_create - n_same

            # create hyperedges as the implanted sets
            for _ in range(n_edges_to_create - n_to_expand):
                interactions.append(g)

            # create hyperedges by adding random nodes to the implanted sets
            other_nodes = list(set(range(N)).difference(g))
            for _ in range(n_to_expand):

                # sample number of additional nodes to add from uniform distribution
                size = rng.choice(range(1, max_size + 1 - len(g)))
                new_edge = g + tuple(rng.choice(other_nodes, replace=False, size=size))
                interactions.append(new_edge)      

        else:
            # if f is None, we create all interactions by adding random nodes to the implanted sets
            # note that the sampling of the size starts from 0

            other_nodes = list(set(range(N)).difference(g))
            for _ in range(n_edges_to_create):
                size = rng.choice(range(0, max_size + 1 - len(g)))
                new_edge = g + tuple(rng.choice(other_nodes, replace=False, size=size))
                interactions.append(new_edge)


    # convert to node-group bipartite representation
    M = N # to index edges starting from N
    edges = []
    for clique in interactions:
        for t in clique:
            edges.append((t, M))
        M += 1

    df = pd.DataFrame(edges, columns=['a', 'b'])
    
    return df, groups








