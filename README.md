# Risk-Averse Matchings over Uncertain Graph Databases 

*Charalampos E. Tsourakakis, Shreyas Sekar, Johnson Lam, Liu Yang*

Contains an installable module, risk-averse-matching, for finding a bounded-variance or bounded-standard deviation matching on a uncertain graph or hypergraph. Also contains module for generating synthetic graphs with the following models and attributes:
- Erdos-Renyi, Barabasi-Albert graph models 
- Bernoulli (weight and probability parameters), Gaussian (mean and variance parameters) distributed edges
- Uniform, Gaussian distributions to sample respective parameters w.r.t. to the edge distribution 

Experiments were done on the following datasets: DBLP citation hypergraph, PPI graph, and synthetically generated graphs. These scripts have also been provided. 

### Setup 

In the `risk-averse-matching/` directory, create and start a virtualenv for the project 
```shell
>>> virtualenv venv
>>> source venv/bin/activate
>>> pip install -e . 
```
To use a virtualenv within jupyter notebook. Run the following command and switch to the "venv" kernel in the notebook. 
```shell 
>>> python -m ipykernel install --user --name=venv
```

### Example: Generating Synthetic Graph 
To generate an Erdos-Renyi graph, Bernoulli distributed edges, Uniformly sampled weights, and Gaussian sampled probabilities
```python 
from risk_averse_matching import hypergraph_matchings as hm 
from risk_averse_matching import graph_generator as gg

graph, edge, weight, prob = 'erdos', 'bernoulli', 'uniform', 'gaussian'
g_param = { 'vertices': 6000, 'p': 0.005 }
w_param = { 'min': 0, 'max': 1000, 'discrete': True }
p_param = { 'mu': 0.5, 'sigma': 0.5/3, 'discrete': False, 'min': 0 }
edge_list = gg.gen_graph_attrib(graph, g_param, edge, weight, w_param, prob, prob_param)
```

### Example: Finding Bounded-Variance Matchings 
Using the maximum matching's variance, find bounded-variance matchings for 20 evenly split intervals
```python
intervals = 20
variance_beta = True 
g = hm.Hypergraph(edge_list, variance_beta, weight='weight', probability='probability', edge='edge', edge_distribution='bernoulli')  
beta_thresholds = g.gen_betas(intervals) 
for idx, beta in enumerate(beta_thresholds):
    bv_matching, bv_stat = g.bounded_matching(beta)
    # save or use returned matching and matching's stats
```

