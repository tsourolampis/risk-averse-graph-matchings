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
    param1 = 'none'
    param2 = ['uniform', 'power', 'inorder', 'inverse']

    results = []
    for e in edges:
        for p2 in param2:
            results.append('{}-{}-{}'.format(e, param1, p2))
    return results

def gen_params(edge_distrib=None, param2_distrib=None):
    p2 = {
        # bernoulli probability parameter
        'bernoulli': {
            'uniform': {'min': 0, 'max': 1, 'discrete': False},
            'power': {'alpha': 2, 'max_int': 1, 'discrete': False},
            'inorder': {},
            'inverse': {}
        },
        # gaussian variance parameter
        'gaussian': {
            'uniform': {'min': 0, 'max': 100, 'discrete': False},
            'power': {'alpha': 2, 'max_int': 50, 'discrete': False},
            'inorder': {},
            'inverse': {}
        }
    }
    param2_vals = p2[edge_distrib][param2_distrib] if param2_distrib else None
    return param2_vals

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
    path = 'data/youtube'
    print('Loading in youtube data...')
    f = '{}/{}'.format(path, 'youtube.pkl')
    graph = pickle.load( open(f, 'rb'))
    print('Starting experiment on {} graph with {} edges'.format(f, len(graph)))

    intervals = 20
    p2_experiments = 5 # number of samples

    total_time = 0
    edge_types = gen_edge_strings() # all combinations of graph parameters
    for e_idx, edge_type in enumerate(edge_types):
        e, p1, p2 = parse(edge_type) # graph parameters
        if e == 'gaussian':
            for edge in graph:
                edge['expected_weight'] = edge['weight']
        for p2_sample in range(p2_experiments):
            # skip 'inorder' and 'inverse' after 1 iteration
            if (p2 == 'inorder' or p2 == 'inverse') and p2_sample > 0:
                break

            start = time.time()
            p2_param = gen_params(edge_distrib=e, param2_distrib=p2)
            graph_p1_p2 = gg.gen_attrib(graph, e, param2_distrib=p2, param2=p2_param)

            print(e_idx, edge_type, p2_sample)
            print('{} edges in synthethic graph. first edge: {}'.format(len(graph_p1_p2), graph_p1_p2[0]))
            p1_attrib = 'weight' if e == 'bernoulli' else 'expected_weight'
            p2_attrib = 'probability' if e == 'bernoulli' else 'variance'
            avg_p1 = sum(e[p1_attrib] for e in graph_p1_p2)/len(graph_p1_p2)
            avg_p2 = sum(e[p2_attrib] for e in graph_p1_p2)/len(graph_p1_p2)
            print('{} avg {} and {} avg {}'.format(avg_p1, p1_attrib, avg_p2, p2_attrib))

            max_stats, bv_stats = run_experiment(graph_p1_p2, intervals, e)
            path = 'data/youtube/results/youtub-{}-{}-{}_{}/'.format(\
                    e, p1, p2, p2_sample)
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
