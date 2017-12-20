from risk_averse_matching import hypergraph_matchings as hm
from risk_averse_matching import graph_generator as gg
import os
import time
import pickle
import copy

def parse(filename):
    return filename.split('-')

def mkdir_subdirec(sub_direc):
    abs_path = os.getcwd()
    full_path = '{}/{}'.format(abs_path, sub_direc)
    os.makedirs(full_path, exist_ok=True)

def gen_edge_strings():
    edges = ['bernoulli', 'gaussian']
    param1 = ['uniform', 'power', 'gaussian']
    param2 = 'none'

    results = []
    for e in edges:
        for p1 in param1:
            results.append('{}-{}-{}'.format(e, p1, param2))
    return results

def gen_params(edge_distrib=None, param1_distrib=None):
    p1 = {
        # bernoulli weight parameter
        'bernoulli': {
            'uniform': {'min': 0, 'max': 1000, 'discrete': True},
            'gaussian': {'mu': 100, 'sigma': 50/3, 'discrete': True, 'min': 0},
            'power': {'alpha': 2, 'max_int': 100, 'discrete': True}
        },
        # gaussian mean parameter
        'gaussian': {
            'uniform': {'min': 0, 'max': 1000, 'discrete': False},
            'gaussian': {'mu': 100, 'sigma': 50/3, 'discrete': False, 'min': 0},
            'power': {'alpha': 2, 'max_int': 1000, 'discrete': False}
        }
    }
    param1_vals = p1[edge_distrib][param1_distrib] if param1_distrib else None
    return param1_vals

def run_experiment(graph, intervals, edge_distrib, path=None, beta_var=False):
    g = hm.Hypergraph(graph, 'probability', 'weight', 'edge', distrib=edge_distrib)
    # maximum matching
    _, max_stat = g.max_matching()
    print('Maximum matching')
    g.print_stats(max_stat)
    # bounded variance matching
    beta_thresholds = g.gen_betas(intervals, beta_var=beta_var)
    bv_results = []
    for idx, beta in enumerate(beta_thresholds):
        if beta_var:
            bv_matching , bv_stat = g.bounded_var_matching(beta, edge_distrib)
        else:
            bv_matching , bv_stat = g.bounded_std_matching(beta, edge_distrib)
        bv_results.append(bv_stat)
        if path is not None:
            f = path + 'bv_matchings-{}.pkl'.format(idx)
            pickle.dump(bv_matching, open(f, 'wb'))
        g.print_stats(bv_stat, beta)
    return max_stat, bv_results

def main():
    path = 'data/ppi'
    f = '{}/{}'.format(path, 'ppi_graph.pkl')
    print('Loading in ppi data...')
    graph_bern = pickle.load( open(f, 'rb'))
    graph_gaus = []
    for edge in graph_bern:
        e_copy = copy.deepcopy(edge)
        e_copy['variance'] = 100 * e_copy['probability']
        e_copy.pop('probability', None)
        graph_gaus.append(e_copy)

    intervals = 20
    p1_experiments = 5 # number of samples
    total_time = 0
    beta_var = True
    edge_types = gen_edge_strings() # all combinations of graph parameters
    print('Starting experiment on {} graph with {} edges'.format(f, len(graph_bern), len(graph_gaus)))
    print(edge_types)
    for e_idx, edge_type in enumerate(edge_types):
        e, p1, p2 = parse(edge_type) # graph parameters
        for p1_sample in range(p1_experiments):
            start = time.time()
            p1_param = gen_params(edge_distrib=e, param1_distrib=p1)
            if e == 'bernoulli':
                graph_p1_p2 = gg.gen_attrib(graph_bern, e, param1_distrib=p1, param1=p1_param)
            elif e == 'gaussian':
                graph_p1_p2 = gg.gen_attrib(graph_gaus, e, param1_distrib=p1, param1=p1_param)


            print(e_idx, edge_type, p1_sample)
            print('{} edges in synthethic graph. first edge: {}'.format(len(graph_p1_p2), graph_p1_p2[0]))
            p1_attrib = 'weight' if e == 'bernoulli' else 'expected_weight'
            p2_attrib = 'probability' if e == 'bernoulli' else 'variance'
            avg_p1 = sum(e[p1_attrib] for e in graph_p1_p2)/len(graph_p1_p2)
            avg_p2 = sum(e[p2_attrib] for e in graph_p1_p2)/len(graph_p1_p2)
            print('{} avg {} and {} avg {}'.format(avg_p1, p1_attrib, avg_p2, p2_attrib))

            path = 'data/ppi/results-variance/ppi-{}-{}-{}_{}/'.format(e, p1, p2, p1_sample) if beta_var else 'data/ppi/results-variance/ppi-{}-{}-{}_{}/'.format(e, p1, p2, p1_sample)
            mkdir_subdirec(path)
            max_stats, bv_stats = run_experiment(graph_p1_p2, intervals, e, beta_var=beta_var)
            print('Finished finding bounded variance matchings')
            f = path + 'max_stats.pkl'
            pickle.dump(max_stats, open(f, 'wb'))
            f = path + 'bv_stats.pkl'
            pickle.dump(bv_stats, open(f, 'wb'))

            t = time.time() - start
            total_time += t
            print('{} sec {} total_time\n'.format(t, total_time))


if __name__ == '__main__':
    main()
