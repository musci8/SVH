from itertools import combinations
import numpy as np
from scipy import special, stats
import pandas as pd



def create_benchmark_OLD(N, T, max_order, closure, max_size=10, n_interactions=20, rng=None):
    """
    Create a synthetic hypergraph benchmark with specified properties.

    Parameters
    -------------
    N: int,
        Number of nodes in the hypergraph.
    T: int,
        Number of sets to create for each order. Note that the actual number of sets created may be higher due to the closure process.
    max_order: int,
        Maximum order of the hyperedges.
    closure: float,
        Fraction of closed hyperedges.
    max_size: int,
        Maximum size of additional interactions.
    n_interactions: int,
        Number of interactions per group.
    rng: np.random.RandomState, optional
        Random number generator for reproducibility. If None, a default generator with seed 42 will be used.
    """

    if rng is None:
        rng = np.random.RandomState(42)
    
    groups = []
    for order in range(2, max_order + 1):

        # when generating sets through closure, each clique of size n+1 generates n+1 groups
        # thus, the number of such cliques to be generated is equal to T / (n+1)
        # this is the reason why we get kk and n_cliques
        #t = int(special.binom(order + 1, order))  # NOTE: this is always equal to order+1
        #kk = int(T / t)
        n_cliques = int(T / (order + 1))
        
        #if kk == 0: 
        if n_cliques == 0:
            raise ValueError(f"Too few sets requested for order {order}. Increase T or decrease max_order.")

        # get the number of sets to be generated via closure
        num_closed = int(closure * n_cliques)
        num_open = n_cliques - num_closed
             
        closed = 0
        all_g = 0
        # create closed simplices
        #for _ in range(int(closure * kk)):
        for _ in range(num_closed):
            #larger_g = tuple(sorted(np.random.choice(range(N), replace=False, size=order+1)))
            clique_nodes = rng.choice(range(N), replace=False, size=order+1)

            # add all the subsets of size n
            for g in combinations(clique_nodes, order):
                groups.append(g)
                closed += 1
                all_g += 1

            # groups.extend(combinations(larger_g, order))
            # closed += t
            # all_g += t

        # create groups that are not closed
        #for _ in range(int((1 - closure) * kk)):
        for _ in range(num_open):
            #for __ in range(t):
            for __ in range(order + 1):
                g = rng.choice(range(N), replace=False, size=order)
                groups.append(tuple(sorted(g)))
                all_g += 1
                    
    # eliminates groups that are proper faces of larger groups
    # i.e. if we have (1, 2) and (1, 2, 3) as groups, we remove (1, 2)
    for order in range(2, max_order + 1):
        larger_g = set()
        all_larger_groups = filter(lambda x: len(x) > order, groups)

        # generate all possible combinations of order 'order' from larger groups
        for l in map(lambda x: tuple(combinations(x, order)), all_larger_groups):
            for g in l: 
                 larger_g.add(g)

        # filter out the smaller groups that are faces of larger ones
        groups = list(filter(lambda x: (x not in larger_g), groups ))

    groups = list(set(groups))  

    # create hyperedges for each of the created groups
    interactions = []
    for g in groups:
        # generate l interactions sampled from binomial
        n_edges_to_create = stats.binom.rvs(p=.5, n=n_interactions)
        for _ in range(n_edges_to_create):

            # sample size from uniform distribution
            size = rng.choice(range(0, max_size + 1 - len(g)))

            # each hyperedge contains the group g plus a random set of other nodes such that whole size is l
            interactions.append(g + tuple(rng.choice(list(set(range(N)).difference(g)), replace=False, size=size)))       

    # convert to node-group bipartite representation
    M = N
    edges = []
    for clique in interactions:
        for t in clique:
            edges.append((t, M))
        M += 1

    df = pd.DataFrame(edges, columns=['a', 'b'])
    
    return df, groups



def create_benchmark(N, T, max_size_implanted_sets, closure, f=1., max_size=10, n_interactions=20, rng=None):
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
    f: float,
        Fraction of interactions in which an implanted set is expanded into a larger hyperedge by adding randomly selected nodes.
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
        n_edges_to_create = stats.binom.rvs(p=.5, n=n_interactions)

        # get the fraction of those to be expanded into larger hyperedges
        n_to_expand = int(f * n_edges_to_create)

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

    # convert to node-group bipartite representation
    M = N # to index edges starting from N
    edges = []
    for clique in interactions:
        for t in clique:
            edges.append((t, M))
        M += 1

    df = pd.DataFrame(edges, columns=['a', 'b'])
    
    return df, groups








