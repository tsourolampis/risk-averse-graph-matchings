# from scipy.stats import bernoulli
import networkx as nx
import numpy as np
import powerlaw as pl
import random

ER = 'erdos'
BA = 'barabasi'
UAR = 'uniform'
PL = 'power'
GAUS = 'gaussian'
BERN = 'bernoulli'

# erdos renyi graph
def __er_graph(n,p,seed=None):
    return nx.fast_gnp_random_graph(n,p,seed=seed)

# barabasi albert graph
def __ba_graph(n, m, seed=None):
    return nx.barabasi_albert_graph(n,m,seed=seed)

# uniform at random distribution
def __uar_param_generator(low, high, size, integer=True, seed=None):
    np.random.seed(seed) # debugging
    if integer:
        return np.random.randint(low, high+1, size)
    return np.random.uniform(low, high, size)

# power law distribution
def __pl_param_generator(alpha, size, max_int=1,integer=False, epsilon=1, seed=None):
    '''
    NOTE: can't seed generator
    '''
    distr = pl.Power_Law(xmax=max_int, parameters=[alpha], discrete=integer)
    s = distr.generate_random(size)
    if max_int == 1:
        maxi = 1
        s = [x/np.float64(100) if x <100 else maxi for x in s]
    return s

def __gaus_param_generator(mu, sigma, size, min_thresh=None, min_val=None, max_thresh=None, max_val=None, seed=None):
    np.random.seed(seed)
    sample = np.random.normal(mu, sigma, size)
    if min_thresh is not None and min_thresh is not None: sample[sample < min_thresh] = min_val
    if max_thresh is not None and max_thresh is not None: sample[sample > max_thresh] = max_val
    return sample

def __bern_generator(weight, weight_params, prob, prob_params, size, edge_list=None, seed=None):
    '''
    Generate weighted bernoulli distributed edges by sampling probs and weights from uniform or power law distribution

   :param str prob: "uniform" or "power"
   :param dict prob_params: w/ keys "min", "max", "discrete"
   :param str weigh: "uniform" or "power"
   :param dict weight_params: w/ keys "alpha", "max_int", "discrete"
   :param int size: number of values to generate
   :param int seed: seed generator
   :return: mean_vals, var_vals
   :rtype: tuple(list, list)
    '''
    try:
        if weight == UAR:
            weight_vals = __uar_param_generator(weight_params['min'], weight_params['max'], size, weight_params['discrete'], seed=seed)
        elif weight == PL:
            weight_vals = __pl_param_generator(weight_params['alpha'], size, weight_params['max_int'], weight_params['discrete'], seed=seed)
        elif weight == GAUS:
            weight_vals = __gaus_param_generator(weight_params['mu'], weight_params['sigma'], size, min_thresh=0, min_val=weight_params['min'])
        else:
            weight_vals = None
    except:
        raise KeyError('Specify weight_params dict w/ correct keys: {}'.format(weight_params))


    try:
        if prob == UAR:
            prob_vals = __uar_param_generator(0, 1, size, prob_params['discrete'], seed)
        elif prob == PL:
            prob_vals = __pl_param_generator(prob_params['alpha'], size, prob_params['max_int'], prob_params['discrete'], seed=seed)
        elif prob == GAUS:
            prob_vals = __gaus_param_generator(prob_params['mu'], prob_params['sigma'], size, min_thresh=0, min_val=prob_params['min'], max_thresh=1, max_val=prob_params['max'])
        elif prob == 'inorder':
            c = 0.5
            # TODO: how to set constant?
            if weight_vals:
                #c = 2 if weight == UAR else 0.5
                total = sum( weight**c for weight in weight_vals) 
                #total = sum(np.sqrt(weight) for weight in weight_vals)
                prob_vals = [weight**c/total for weight in weight_vals]
            elif not weight_vals and edge_list:
                total = sum(edge['weight']**c) for edge in edge_list)
                prob_vals = [edge['weight']**c/total for edge in edge_list]
            else:
                raise('generating only "inorder" probability requires an edge_list')

        elif prob == 'inverse':
            if weight_vals:
                #c = 2 if weight == UAR else 0.5
                total = sum(weight**c for weight in weight_vals)
                prob_vals = [(1 - (weight**c/total)) for weight in weight_vals]
            elif not weight_vals and edge_list:
                total = sum(edge['weight']**c for edge in edge_list)
                prob_vals = [(1- (edge['weight']**c/total)) for edge in edge_list]
            else:
                raise('generating ONLY "inverse" probability requires an edge_list')

        else:
            prob_vals = None
    except:
        raise KeyError('Specify prob_params dict w/ correct keys: {} {}'.format(prob, prob_params))

    return weight_vals, prob_vals

def __gaus_generator(mean, mean_params, var, var_params, size, edge_list=None, seed=None):
    '''
    Generate gaussian distributed edges by sampling means and variances from uniform or power law distribution

    :param str mean: "uniform" or "power"
    :param dict mean_params: w/ keys "min", "max", "discrete"
    :param str var: "uniform" or "power"
    :param dict var_params: w/ keys "alpha", "max_int", "discrete"
    :param int size: number of values to generate
    :param int seed: seed generator
    :return: mean_vals, var_vals
    :rtype: tuple(list, list)
    '''
    try:
        if mean == UAR:
            mean_vals = __uar_param_generator(mean_params['min'], mean_params['max'],  size, mean_params['discrete'], seed)
        elif mean == PL:
            mean_vals = __pl_param_generator(mean_params['alpha'], size, mean_params['max_int'], mean_params['discrete'], seed=seed)
        elif mean == GAUS:
            mean_vals = __gaus_param_generator(mean_params['mu'], mean_params['sigma'], size, min_thresh=0, min_val=mean_params['min'])
        else:
            mean_vals = None
    except:
        raise KeyError('Specify mean_params dict w/ correct keys: {}'.format(mean_params))


    try:
        if var == UAR:
            var_vals = __uar_param_generator(var_params['min'], var_params['max'], size, var_params['discrete'], seed)
        elif var == PL:
            var_vals = __pl_param_generator(var_params['alpha'], size, var_params['max_int'], var_params['discrete'], seed=seed)
        elif var == GAUS:
            var_vals = __gaus_param_generator(var_params['mu'], var_params['sigma'], size, min_thresh=0,min_val=var_params['min'])
        elif var == 'inorder':
            # TODO: how to set constant?
            if mean_vals:
                var_vals = [0.5*mu for mu in mean_vals]
            elif not mean_vals and edge_list:
                var_vals = [0.5*edge['expected_weight'] for edge in edge_list]
            else:
                raise('generating ONLY "inorder" variance requires an edge_list')

        elif var == 'inverse':
            if mean_vals:
                constant = max(mean_vals)
                var_vals = [0.5*(constant-mu) if mu < constant else 0 for mu in mean_vals]
            elif not mean_vals and edge_list:
                constant = max(edge_list, key=lambda e: e['expected_weight'])['expected_weight']
                var_vals = [0.5*(constant - edge['expected_weight']) for edge in edge_list]
            else:
                raise('generating ONLY "inverse" variance requires an edge_list')

        else:
            var_vals = None
    except:
        raise KeyError('Specify var_params dict w/ correct keys: {}'.format(var_params))

    return mean_vals, var_vals

def gen_graph(graph, graph_params, seed=None):
    '''
    Generate only a graph give a graph type
    '''
    try:
        vertices, graph_prob = graph_params['vertices'], graph_params['p']
        if graph == ER:
            g = __er_graph(vertices, graph_prob, seed)
        else:
            g = __ba_graph(vertices, int(vertices*graph_prob)//2, seed) # (p*n)/2
    except:
        raise ValueError('specify graph_params w/ keys "vertices" and "p"')
    edge_list = list(g.edges())
    random.shuffle(edge_list)
    edge_list = [{'authors':[v1,v2]} for v1, v2 in edge_list]
    return edge_list

def gen_attrib(edge_list, edge_distrib=BERN, param1_distrib=None, param1=None, param2_distrib=None, param2=None, seed=None):
    '''
    Generate edge attributes given a graph and parameters of the edge attribute
    '''

    edge_count = len(edge_list)
    if edge_distrib == BERN:
        weight_vals, prob_vals = __bern_generator(param1_distrib, param1, param2_distrib, param2, edge_count, edge_list, seed)
        for idx, edge in enumerate(edge_list):
            if param1_distrib: edge['weight'] = weight_vals[idx]
            if param2_distrib: edge['probability'] = prob_vals[idx]
            edge_list[idx] = edge

    elif edge_distrib == GAUS:
        mean_vals, var_vals = __gaus_generator(param1_distrib, param1, param2_distrib, param2, edge_count, edge_list, seed)
        print(param1_distrib, param1, param2_distrib, param2, edge_count, seed)
        for idx, edge in enumerate(edge_list):
            if param1_distrib: edge['expected_weight'] = mean_vals[idx]
            if param2_distrib: edge['variance'] = var_vals[idx]
            edge_list[idx] = edge
    else:
        raise ValueError('Specify a "bernoulli" or "gaussin" distribution for the edge')

    return edge_list


def gen_graph_attrib(vertices, graph_prob, graph=ER, edge_distrib=BERN, param1_distrib=UAR, param1=None, param2_distrib=UAR, param2=None, assign='random', seed=None):
    '''
    Generate a graph and its edge attributes
    '''
    g = None
    if graph == ER:
        g = __er_graph(vertices, graph_prob, seed)
    else:
        g = __ba_graph(vertices, int(vertices*graph_prob), seed) # 10 edges per node
    edge_list = list(g.edges())
    edge_count = len(g.edges())
    random.shuffle(edge_list)

    # TODO: change edge key 'authors' to 'edge'
    if edge_distrib == BERN:
        weight_vals, prob_vals = __bern_generator(param1_distrib, param1, param2_distrib, param2, edge_count, seed)
        for idx, edge in enumerate(edge_list):
            e = {
                'authors':list(edge),
                'probability': prob_vals[idx],
                'weight': weight_vals[idx]
            }
            edge_list[idx] = e

    elif edge_distrib == GAUS:
        mean_vals, var_vals = __gaus_generator(param1_distrib, param1, param2_distrib, param2, edge_count, seed)
        for idx, edge in enumerate(edge_list):
            e = {
                'authors':list(edge),
                'expected_weight': mean_vals[idx],
                'variance': var_vals[idx]
            }
            edge_list[idx] = e
    else:
        raise ValueError('Specify a "bernoulli" or "gaussin" distribution')

    return edge_list

