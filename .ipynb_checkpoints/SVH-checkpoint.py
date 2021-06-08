from rpy2.robjects.packages import importr
from rpy2.robjects import r,IntVector
from rpy2.robjects.vectors import ListVector
from collections import OrderedDict
import pandas as pd
from multiprocessing import Pool,cpu_count
from itertools import combinations
from scipy.special import binom
import scipy.stats as st
import numpy as np
import igraph
importr('SuperExactTest')

def _p_over(t):
    w,n,na,nb = t
    return st.hypergeom.sf(w-1,n,na,nb)
    

def _pvalue_intersect(X):
    t,neighs,N = X
    lists = [neighs[node] for node in t]
    rlists = [IntVector(l) for l in lists]  
    inters = set(lists[0])
    for n in lists[1:]: inters = inters.intersection(n)
    inters = len(inters)  
    lengths = sorted(map(len,lists))
    d = OrderedDict(zip(map(str, range(len(rlists))), rlists))
    data = ListVector(d)
    res = r['supertest'](data,n=N)
    return list(dict(zip(res.names, list(res)))['P.value'])[-1]
    
    
def create_bench(n_bench,m_per_group,scale,noise):
	"""
	Create a synthetic hypergraph.

	Parameters
	-------------
	n_bench:		int
		Number of nodes
		
	m_per_groups:	int
		Weight of each hyperlink
		
	scale:			float
		Fraction of nodes that participate to a hyperlink at each size
		
	noise:			float
		Probability of creating hyperlinks among random groups

	Returns
	-------------
	df:				DataFrame 
		Table with two columns ['a','b'] that represent the generated hypergraph. 'a' contains the nodes of the hypergraph and 'b' the hyperlink a node participate to
		
	groups_bench:	dict
		Dictionary where keys are size and values the lists of generated hyperlinks
	"""

	edges = []
	groups_bench = {}
	M = n_bench

	for i,order in enumerate(range(2,10)):

		n_groups = int(scale*n_bench/order)

		groups_bench[order] = np.random.choice(range(n_bench),size=(n_groups,order),replace=False)
		
		for group in groups_bench[order]:
			for _ in range(m_per_group):
				if np.random.rand()<(1-noise):
					edges.extend([(node,M) for node in group])
				else:
					rand_group = np.random.choice(range(n_bench),size=order,replace=False)
					edges.extend([(node,M) for node in rand_group])
				M+=1  
				
	df = pd.DataFrame(edges,columns=['a','b'])
			   
	return df,groups_bench

def get_svh(df,max_size=10,mp=True):
	"""
	Extract the Statistically Validated Hypergraph.

	Parameters
	-------------
	df:			DataFrame 
		Table with two columns ['a','b'] that represent the hypergraph to be validated. 'a' contains the nodes of the hypergraph and 'b' the hyperlink a node participate to.
		
	max_size:	int
		Maximum size of the hyperlinks to be tested
		
	mp:			Bool (default: True)
		Specify whether to use multithreading or not. If True, it will use all available cores.
		
	Returns
	-------------
	svh:		dict
		Dictionary where keys are hyperlink size and values are DataFrame with the result of validation. Each DataFrame is a Table with columns ['group','pvalue','fdr']. 
		'group' contains all the hyperlinks (mapped as tuples) present in the hypergraph
		'pvalue' reports the pvalue
		'fdr' is a bool that is True if the hyperlink belongs to the SVH, False otherwise
	"""

	deg_set_b = df.groupby('b')['a'].count().reset_index()

	orders = deg_set_b.a.unique()
	orders = orders[(orders>=2)&(orders<=max_size)]
	pvalues = {}

	for order in np.sort(orders):
		sub_deg = deg_set_b.query('a==@order').b.tolist()
		sub_edges = df.query('b in @sub_deg')
		tuples = sub_edges.groupby('b')['a'].apply(lambda x: tuple(sorted(x))).unique().tolist()
		tuples_order = list(filter(lambda x: len(x)==order,tuples))
		neigh_set_a_sub = dict(sub_edges.groupby('a')['b'].apply(list).reset_index().values)
		N = len(sub_deg)


		if mp:
			p = Pool(processes=cpu_count())
			pvalues[order] = dict(zip(tuples_order,p.map(_pvalue_intersect,zip(tuples_order,[neigh_set_a_sub]*len(tuples_order),[N]*len(tuples_order)))))
			p.close()
		else:
			pvalues[order] = dict(zip(tuples_order,map(_pvalue_intersect,zip(tuples_order,[neigh_set_a_sub]*len(tuples_order),[N]*len(tuples_order)))))

	svh = {}
	links = 0
	links_order = {}
	for order in sorted(pvalues):
		n_a = len(set(np.concatenate(list(pvalues[order].keys()))))
		n_possible = binom(n_a,order)
		bonf = 0.01/n_possible

		temp_df = pd.DataFrame(pvalues[order].items())
		temp_df.columns = ['group','pvalue']
		ps = np.sort(temp_df.pvalue)
		k = np.arange(1,len(ps)+1)*bonf
		try: fdr = k[ps<k][-1] 
		except: fdr = 0
		temp_df['fdr'] = temp_df['pvalue']<fdr
		svh[order] = temp_df
		
	return svh



def get_svn(df,mp=True):
	"""
	Extract the Statistically Validated Network.

	Parameters
	-------------
	df:			DataFrame 
		Table with two columns ['a','b'] that represent the hypergraph to be validated. 'a' contains the nodes of the hypergraph and 'b' the hyperlink a node participate to
		
	mp:			Bool (default: True)
		Specify whether to use multithreading or not. If True, it will use all available cores
		
	Returns
	-------------
	svn:		Graph object (igraph)
		graph object with the pairwise links that belong to the SVN

	"""

	g = igraph.Graph.TupleList(df.values)
	set_b = dict(zip(df.b.unique(),[1]*df.b.nunique()))
	names = g.vs['name']
	g.vs['type'] = [set_b.get(names[i],0) for i in range(g.vcount())]    
	g_deg = dict(zip(names,g.degree()))
	nb = df['b'].nunique()

	g_bip = g.bipartite_projection(which=0,multiplicity=True)
	names = g_bip.vs['name']

	tuples = []
	for i,w in enumerate(g_bip.es['weight']):
		edge = g_bip.es[i]
		s = edge.source
		t = edge.target
		tuples.append((w,nb,g_deg[names[s]],g_deg[names[t]]))

	g_bip.es['params'] = tuples

	tuples = sorted(set(g_bip.es['params']))

	if mp:
		p = Pool(processes=cpu_count())
		pvalues = dict(zip(tuples,p.map(_p_over,tuples)))
		p.close()
	else:
		pvalues = dict(zip(tuples,map(_p_over,tuples)))

	g_bip.es['pvalues'] = [pvalues[i] for i in g_bip.es['params']]

	a,b = list(zip(*g_bip.get_edgelist()))

	a = list(map(lambda x: names[x],a))
	b = list(map(lambda x: names[x],b))

	ps = pd.DataFrame(zip(a,b,g_bip.es['pvalues']),columns=['source','target','p'])

	ps['test'] = 'fail'
	pvalues = np.sort(ps['p'])

	n = df.a.nunique()
	nt = n*(n-1)/2

	ks = np.arange(1,len(pvalues)+1)*0.01/nt

	try:
		fdr = ks[pvalues<=ks][-1]
	except IndexError:
		fdr = 0

	ps.loc[ps.p<fdr,'test'] = 'success'
	ps['weight'],ps['N'],ps['Na'],ps['Nb'] = list(zip(*g_bip.es['params']))

	svn = igraph.Graph(list(map(tuple,ps.query('test=="success"')[['source','target']].values)))
		
	return svn


def compute_tpr(svh,svn,groups_bench):
	"""
	Compute the True Positive Rate and False Discovery Rate for the SVH and SVN extracted from a syntethic benchmark

	Parameters
	-------------
	svh:		dict
		Dictionary where keys are hyperlink size and values are DataFrame with the result of validation. Each DataFrame is a Table with columns ['group','pvalue','fdr']. 
		'group' contains all the hyperlinks (mapped as tuples) present in the hypergraph
		'pvalue' reports the pvalue
		'fdr' is a bool that is True if the hyperlink belongs to the SVH, False otherwise
		
	svn: 		Graph object (igraph)	
		graph object with the pairwise links that belong to the SVN
		
	groups_bench:	dict
		Dictionary where keys are size and values the lists of true hyperlinks
		
	Returns
	-------------
	results:	dict
		Nested dictionary where first level of keys is related to measures (TPR svh, TPR svn, FDR svh, FDR svn) and second level is related to the hyperlink size
	"""

	TPR_ho = {}
	TPR_pw = {}

	FDR_ho = {}
	FDR_pw = {}

	svn_cliques = list(map(lambda x: tuple(sorted(x)), svn.maximal_cliques()))

	for order in range(3,9):
		
		temp = svh[order].query('fdr')
		tuples_ho = set(list(map(lambda x: tuple(sorted(x)),temp.group)))
		tuples_real = set(list(map(lambda x: tuple(sorted(x)),groups_bench[order])))
		tuples_pw = set(list(filter(lambda x: len(x)==order,svn_cliques)))
		
		TPR_ho[order] = len(tuples_real.intersection(tuples_ho))/len(tuples_real)
		TPR_pw[order] = len(tuples_real.intersection(tuples_pw))/len(tuples_real)
		
		try: FDR_ho[order] = len(tuples_ho.difference(tuples_real))/len((tuples_ho))
		except: FDR_ho[order] = np.nan
		try: FDR_pw[order] = len(tuples_pw.difference(tuples_real))/len((tuples_pw))
		except: FDR_pw[order] = np.nan
		
	results = {'TPR_ho':TPR_ho,
			'TPR_pw':TPR_pw,
			'FDR_ho':FDR_ho,
			'FDR_pw':FDR_pw
			}

	return results
