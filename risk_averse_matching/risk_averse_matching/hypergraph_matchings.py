from collections import defaultdict
import numpy as np
import time
import warnings

# arbitrary constant
MAX_MATCHING = float('inf')
np.seterr(all='raise')
warnings.filterwarnings('error')

class Hypergraph:
    def __init__(self, data, prob=None, weight=None, \
                    exp_weight='expected_weight', alpha='alpha', \
                    var='variance', std='standard_deviation', \
                    distrib='bernoulli', epsilon=0.01\
                ):
        '''
        Initializes values and variables needed for finding matchings
        '''
        # TODO: specify edge key
        if not (prob or weight):
            raise ValueError("Please indicate the probability and weight key in dict")
        # edge dictionary keys
        # TODO: clean up -> convert into a dict
        self._weight = weight
        self._prob = prob
        self._exp_weight = exp_weight
        self._alpha = alpha
        self._var = var
        self._std = std
        self._distrib = distrib
        self._epsilon = epsilon

        # Generate expected weight, standard deviation, and alpha
        data = self.__init_attributes(data, distrib)

        # data structures
        self.adj_list = defaultdict(lambda: defaultdict(int))
        self.alpha_sorted = sorted(data, key=lambda d: (d[alpha], -d[prob]), reverse = True)
        self.exp_weight_sorted = sorted(data, key=lambda d: d[exp_weight], reverse = True)

        # store max matching in memory
        # TODO: remove after testing is finished
        self.mmatching = None

    def __init_attributes(self, edges, distrib):
        '''
        generate edge attributes based on distribution of edge
        '''
        # given mean (exp_weight) and variance, generate alpha and standard dev
        if distrib == 'gaussian':
            for entry in edges:
                entry[self._weight] = 0
                entry[self._prob] = 0
                if entry[self._var] != 0:
                    entry[self._alpha] = entry[self._exp_weight]/np.sqrt(entry[self._var])
                else:
                    entry[self._alpha] = entry[self._exp_weight]/np.sqrt(self._epsilon)
                entry[self._std] = self.calc_standard_dev([entry], distrib)
        # given weight and probability, generate alpha, exp weight, and standard dev
        else:
            for entry in edges:
                w = entry[self._weight] if entry[self._weight] > 0 else self._epsilon
                if entry[self._prob] == 0:
                    p = self._epsilon
                elif entry[self._prob] == 1:
                    p = 1 - self._epsilon
                else:
                    p = entry[self._prob]
                try:
                    std = w * np.sqrt(p * (1 - p)) # std = w(sqrt(p(1-p)))
                    entry[self._alpha] = entry[self._weight] * entry[self._prob] / std # alpha = wp / std
                    entry[self._exp_weight] = entry[self._prob] * entry[self._weight]
                    entry[self._std] = self.calc_standard_dev([entry], distrib)
                except FloatingPointError:
                    raise('Error w/ attribute values. Cannot calculate standard deivation and/or alpha: {}'.format(entry))
        return edges

    def print_stats(self, stats, threshold=None):
        if threshold:
            print('{} Beta Threshold: {} edges, {} weight, {} avg probability, {} exp_weight, {} std, {} time'.format(\
                stats['beta'], stats['edges'], stats['weight'],round(stats['probability']/stats['edges'], 2), \
                round(stats['expected_weight'], 2), round(stats['std'], 2) ,round(stats['runtime'], 2)))
        else:
            print('{} edges, {} weight, {} avg probability, {} exp_weight, {} std, {} time'.format(\
                stats['edges'], stats['weight'],  round(stats['probability']/stats['edges'], 2), \
                round(stats['expected_weight'], 2), round(stats['std'], 2) ,round(stats['runtime'], 2)))


    def __add_adj_list(self, edges=None):
        '''
        Update adjacency list with specified additional edges (larger subgraph).
        If edges not specified, update adjacency list with all edges (entire graph)
        @params:
            edges: list of edges to update adjacency list with
        '''
        if edges is None:
            if len(self.adj_list) > 0:
                raise ValueError('Please delete all edges from adjacency list before adding all edges')
            edges = self.exp_weight_sorted

        for edge in edges:
            for author in edge['authors']:
                self.adj_list[author]['nodes'] += 1
                self.adj_list[author]['matched'] = False

    def __del_adj_list(self, edges=None):
        '''
        Update adjcancy list by removing specified edges (smaller subgraph)
        If edges not specified, remove all edges from adjacency list
        @params:
            edges: list of edges to remove from adjacency list
        '''
        if edges is None:
            self.adj_list = defaultdict(lambda: defaultdict(int))
            return
        for edge in edges:
            for author in edge['authors']:
                self.adj_list[author]['nodes'] -= 1
                if self.adj_list[author]['nodes'] == 0 :
                    self.adj_list.pop(author)

    def gen_betas(self, intervals):
        '''
        Generate evenly spaced intervals based on the standard deviation
        @params:
            intervals: Number of evenly spaced intervals
        @returns:
            threshold_vals: list of (beta) threshold values
        '''
        threshold_vals = None
        _, stats = self.max_matching()
        maxi = int(np.ceil(stats['std']))
        mini = 0
        # mini = maxi//intervals
        threshold_vals = [round(val) for val in np.linspace(mini,maxi,intervals+1)]
        print('Generating beta thresholds: {}'.format(threshold_vals))
        return threshold_vals

    def __greedy_matching(self, min_alpha, total_edges=None, threshold=None, distrib='bernoulli'):
        # stored max matching returned if already found
        # TODO: delete after testing finished
        if min_alpha == MAX_MATCHING and self.mmatching is not None:
            return self.mmatching
        # greedy matching statistics
        total_weight = 0
        total_prob = 0
        total_exp_weight = 0
        total_std = 0

        matching_edges = []
        vertex_removed = []
        edge_count = 0
        for e in self.exp_weight_sorted:
            # skip edges with alpha < min alpha of current subgraph
            # TODO: possible bug, < or <=  min_alpha. I think it should be <=
            if min_alpha != MAX_MATCHING and e[self._alpha] <= min_alpha:
                continue

            edge_count += 1
            # skip edge if its standard dev is greater than the current threshold
            # edge_std = self.calc_standard_dev([e], distrib)
            edge_std = e[self._std]
            if threshold and edge_std > threshold:
                continue

            # check if valid edge: all authors for hyperedge are still available
            available = True
            for author in e['authors']:
                if self.adj_list[author]['matched']:
                    available = False
                    break
            # edge is available for matching
            if available:
                total_weight += e[self._weight]
                total_prob += e[self._prob]
                total_exp_weight += e[self._exp_weight]
                total_std += edge_std
                matching_edges.append(e)
                # flag vertices that should not be considered
                for author in e['authors']:
                    self.adj_list[author]['matched'] = True
                    vertex_removed.append(author)
            # breaks when all valid edges in subgraph have been considered
            # TODO: possible bug here
            if total_edges and edge_count == total_edges:
                break

        # reset adjacency list
        for author in vertex_removed:
            self.adj_list[author]['matched'] = False

        # stores max matching to be reused later
        # TODO: delete after testing finished
        if min_alpha == MAX_MATCHING and self.mmatching is None:
            self.mmatching = matching_edges, len(matching_edges), total_weight, total_prob, total_exp_weight, total_std

        return matching_edges, len(matching_edges), total_weight, total_prob, total_exp_weight, total_std

    def bounded_var_matching(self, threshold, distrib='bernoulli'):
        '''
        For a give beta (threshold), find a bounded-variance matching using binary search.
        @params:
            threshold: (pseudo) standard deviation beta.
        @returns:
            matching: greedy matching found for a given gamma and beta
            stats: dictionary of greedy matching statistics
        '''

        start = time.time()

        # initialize variables
        hi = len(self.alpha_sorted) - 1
        lo = 0
        mid = (hi + lo)//2
        self.__del_adj_list()
        self.__add_adj_list(self.alpha_sorted[:mid])

        # Binary Search
        while True:
            # print('low {} mid {} high {}'.format(lo, mid, hi))
            min_alpha = self.alpha_sorted[mid][self._alpha]
            greedy_matching = self.__greedy_matching(min_alpha, total_edges=mid, threshold=threshold, distrib=distrib)
            std = greedy_matching[5] # beta (standard deviation)

            if hi <= mid or lo >= mid:
                break
            elif threshold - std < 0.1 and threshold - std > 0:
                break
            elif std < threshold:
                lo = mid
                mid = (hi+lo)//2
                self.__add_adj_list(self.alpha_sorted[lo:mid])
            else:
                hi = mid
                mid = (hi+lo)//2
                self.__del_adj_list(self.alpha_sorted[mid:hi])
        total_time = time.time() - start
        matching, stats = self.gen_stats_dict(greedy_matching, total_time, threshold)

        #TODO: messy implementation of max output, fix it
        e_next = self.alpha_sorted[mid]
        e_next_stats = [e_next], 1, e_next[self._weight], e_next[self._prob], e_next[self._exp_weight], e_next[self._std]
        _, e_next_stats = self.gen_stats_dict(e_next_stats, 0, threshold)
        return max((matching, stats), ([e_next], e_next_stats), key=lambda x: x[1][self._exp_weight])

    def max_matching(self):
        '''
        Find a greedy matching on the entire graph
        @returns:
            matching: maximum greedy matching
            stats : dictionary of maximum greedy matching statistics
        '''
        # initialize variables
        self.__del_adj_list()
        self.__add_adj_list()

        start = time.time()
        greedy_matching = self.__greedy_matching(MAX_MATCHING)
        total_time = time.time() - start
        matching, stats = self.gen_stats_dict(greedy_matching, total_time)

        return matching, stats

    def gen_stats_dict(self, greedy_matching, total_time, beta=None):
        '''
        Create a dictionary for stats generated from greedy matching
        @params:
            greedy_matching: result returned from __greedy_matching()
            total_time: runtime to run __greedy_matching()
            bv_input: provide (gamma, beta) tuple for a bounded-variance matching
        '''
        matching, size, weight, prob, exp_weight, std = greedy_matching
        stats = {
                'edges': size,
                'weight': weight,
                'probability': prob,
                'expected_weight': exp_weight,
                'std': std,
                'runtime': total_time
        }
        if beta is not None:
            stats['beta'] = beta
        return matching, stats

    # Standard deviation
    def calc_standard_dev(self, edges, distrib):
        if distrib == 'gaussian':
            return sum(np.sqrt(e[self._var]) for e in edges) # sqrt(variance)
        return sum(e[self._weight] * np.sqrt(e[self._prob] * (1-e[self._prob])) for e in edges) # w(sqrt(p(1-p)))
