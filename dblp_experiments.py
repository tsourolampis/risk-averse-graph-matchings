from risk_averse_matching import hypergraph_matchings as hm
import os
import time
import pickle
import json

def mkdir_subdirec(sub_direc):
    abs_path = os.getcwd()
    full_path = '{}/{}'.format(abs_path, sub_direc)
    os.makedirs(full_path, exist_ok=True)

def run_experiment(graph, intervals, edge_distrib, path=None, beta_var=False):
    g = hm.Hypergraph(graph, variance_beta=beta_var, weight='weight', probability='probability', edge='authors', edge_distribution='bernoulli')
    print(g)

    # print('Maximum matching')
    # _, max_stat = g.max_matching()
    # g.print_stats(max_stat)
    max_stat = None


    print('Bounded variance matchings')
    beta_thresholds = g.gen_betas(intervals)
    bv_results = []
    for idx, beta in enumerate(beta_thresholds):
        bv_matching, bv_stat = g.bounded_matching(beta)
        bv_results.append(bv_stat)
        if path is not None:
            f = path + 'bv_matchings-{}.pkl'.format(idx)
            pickle.dump(bv_matching, open(f, 'wb'))
        g.print_stats(bv_stat, beta)
        print(bv_matching)
        print(json.dumps(bv_stat, indent=4))
        break
    return max_stat, bv_results

def main():
    path = 'data/dblp'
    print('Loading in dblp data...')
    # f = '{}/{}'.format(path, 'dblp_v8_graph.pkl')
    f = '{}/{}'.format(path, 'dblp_v10_graph.pkl')
    graph = pickle.load( open(f, 'rb'))
    print('Starting experiment on {} graph with {} hyperedges'.format(f, len(graph)))

    start = time.time()
    edge_distrib = 'bernoulli'
    # edge_distrib = 'gaussian'
    intervals = 20
    beta_var=False
    # path = path + '/results-v8-variance/' if beta_var else path + '/results-v8-standard-deviation/'
    path = path + '/results-v10-variance/' if beta_var else path + '/results-v10-standard-deviation/'
    mkdir_subdirec(path)
    max_stats, bv_stats = run_experiment(graph, intervals, edge_distrib, beta_var=beta_var)

    print('Finished finding bounded variance matchings')
    f = path + 'max_stats.pkl'
    pickle.dump(max_stats, open(f, 'wb'))
    f = path + 'bv_stats.pkl'
    pickle.dump(bv_stats, open(f, 'wb'))


    t = time.time() - start
    print('{} total_time\n'.format(t))


if __name__ == '__main__':
    main()
