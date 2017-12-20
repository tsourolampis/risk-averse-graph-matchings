from risk_averse_matching import hypergraph_matchings as hm
from risk_averse_matching import graph_generator as gg
import os
import time
import pickle

def parse(filename):
    return filename.split('-')

def mkdir_subdirec(sub_direc):
    abs_path = os.getcwd()
    full_path = '{}/{}'.format(abs_path, sub_direc)
    os.makedirs(full_path, exist_ok=True)

def gen_edge_strings():
    edges = ['bernoulli', 'gaussian']
    param1 = ['uniform', 'power', 'gaussian']
    param2 = ['uniform', 'power', 'gaussian', 'inorder', 'inverse']

    results = []
    for e in edges:
        for p1 in param1:
            for p2 in param2:
                results.append('{}-{}-{}'.format(e, p1, p2))
    return results

def gen_params(edge_distrib=None, param1_distrib=None, param2_distrib=None):
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
    p2 = {
        # bernoulli probability parameter
        'bernoulli': {
            'uniform': {'min': 0, 'max': 1, 'discrete': False},
            'power': {'alpha': 2, 'max_int': 1, 'discrete': False},
            'gaussian': {'mu': 0.5, 'sigma': 0.5/3, 'discrete': False, 'min': 0, 'max': 1},
            'inorder': {},
            'inverse': {}
        },
        # gaussian variance parameter
        'gaussian': {
            'uniform': {'min': 0, 'max': 100, 'discrete': False},
            'gaussian': {'mu': 50, 'sigma': 25/3, 'discrete': False, 'min': 0},
            'power': {'alpha': 2, 'max_int': 50, 'discrete': False},
            'inorder': {},
            'inverse': {}
        }
    }
    param1_vals = p1[edge_distrib][param1_distrib] if param1_distrib else None
    param2_vals = p2[edge_distrib][param2_distrib] if param2_distrib else None
    return param1_vals, param2_vals

def run_experiment(graph, intervals, edge_distrib):
    g = hm.Hypergraph(graph, 'probability', 'weight', distrib=edge_distrib)
    # maximum matching
    _, max_stat = g.max_matching()
    print('Maximum matching')
    g.print_stats(max_stat)
    # bounded variance matching
    beta_thresholds = g.gen_betas(intervals)
    bv_results = []
    for beta in beta_thresholds:
        _, bv_stat = g.bounded_var_matching(beta, edge_distrib)
        bv_results.append(bv_stat)
        g.print_stats(bv_stat, beta)
    return max_stat, bv_results

def main():
    path = 'data/wikipedia'
    print('Loading in wikipedia data...')
    f = '{}/{}'.format(path, 'wikipedia.pkl')
    graph = pickle.load( open(f, 'rb'))
    print('Starting experiment on {} graph with {} edges'.format(f, len(graph)))

    intervals = 20
    p1_experiments = 3 # number of samples
    p2_experiments = 3 # number of samples

    total_time = 0
    edge_types = gen_edge_strings() # all combinations of graph parameters
    for e_idx, edge_type in enumerate(edge_types):
        e, p1, p2 = parse(edge_type) # graph parameters
        for p1_sample in range(p1_experiments):
            p1_param, _ = gen_params(edge_distrib=e, param1_distrib=p1)
            graph_p1 = gg.gen_attrib(graph, e, param1_distrib=p1, param1=p1_param)
            for p2_sample in range(p2_experiments):
                # skip 'inorder' and 'inverse' after 1 iteration
                if (p2 == 'inorder' or p2 == 'inverse') and p2_sample > 0:
                    break

                start = time.time()
                _, p2_param = gen_params(edge_distrib=e, param2_distrib=p2)
                graph_p1_p2 = gg.gen_attrib(graph_p1, e, param2_distrib=p2, param2=p2_param)

                print(e_idx, edge_type, p1_sample, p2_sample)
                print('{} edges in synthethic graph. first edge: {}'.format(len(graph_p1_p2), graph_p1_p2[0]))
                p1_attrib = 'weight' if e == 'bernoulli' else 'expected_weight'
                p2_attrib = 'probability' if e == 'bernoulli' else 'variance'
                avg_p1 = sum(e[p1_attrib] for e in graph_p1_p2)/len(graph_p1_p2)
                avg_p2 = sum(e[p2_attrib] for e in graph_p1_p2)/len(graph_p1_p2)
                print('{} avg {} and {} avg {}'.format(avg_p1, p1_attrib, avg_p2, p2_attrib))

                max_stats, bv_stats = run_experiment(graph_p1_p2, intervals, e)
                path = 'data/wikipedia/results/wikipedia-{}-{}-{}_{}{}/'.format(\
                        e, p1, p2, p1_sample, p2_sample)
                print('Finished finding bounded variance matchings')
                mkdir_subdirec(path)
                f = path + 'max_stats.pkl'
                pickle.dump(max_stats, open(f, 'wb'))
                f = path + 'bv_stats.pkl'
                pickle.dump(bv_stats, open(f, 'wb'))

                t = time.time() - start
                total_time += t
                print('{} sec {} total_time\n'.format(t, total_time))


if __name__ == '__main__':
    main()
