from risk_averse_matching import hypergraph_matchings as hm
import os
import time
import pickle

def mkdir_subdirec(sub_direc):
    abs_path = os.getcwd()
    full_path = '{}/{}'.format(abs_path, sub_direc)
    os.makedirs(full_path, exist_ok=True)

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
            bv_matching, bv_stat = g.bounded_var_matching(beta, edge_distrib)
        else:
            bv_matching, bv_stat = g.bounded_std_matching(beta, edge_distrib)
        bv_results.append(bv_stat)
        if path:
            f = path + 'bv_matchings-{}.pkl'.format(idx)
            pickle.dump(bv_matching, open(f, 'wb'))
        g.print_stats(bv_stat, beta)
    return max_stat, bv_results

def main():
    path = 'data/ppi_unweighted'
    print('Loading in ppi unweighted data...')
    f = '{}/{}'.format(path, 'ppi_unweighted_graph.pkl')
    graph = pickle.load( open(f, 'rb'))
    print('Starting experiment on {} graph with {} hyperedges'.format(f, len(graph)))

    start = time.time()
    edge_distrib = 'bernoulli'
    intervals = 20
    beta_var = True
    path = '{}/results-variance/'.format(path) if beta_var else '{}/results-standard-deviation/'.format(path)
    mkdir_subdirec(path)
    max_stats, bv_stats = run_experiment(graph, intervals, edge_distrib, path, beta_var=beta_var)
    print('Finished finding bounded variance matchings')
    f = path + 'max_stats.pkl'
    pickle.dump(max_stats, open(f, 'wb'))
    f = path + 'bv_stats.pkl'
    pickle.dump(bv_stats, open(f, 'wb'))


    t = time.time() - start
    print('{} total_time\n'.format(t))


if __name__ == '__main__':
    main()
