# Risk Averse Team Formations 
Contains an installable module, hypergraphs, for finding a bounded-variance matching on a standard graph or hypergraph and generating synthetic graphs, edge weight, and edge probabilities. Example code is provided to run experiments on a graph and generate synthetic graphs and edge properties. Includes other stand-alone scripts for parsing raw text and xml graph files, generating edge probabilities (as described in the paper), obtaining graph edge and vertex counts, and
wget/curling large files from google drive from the terminal.

### Setup 
In the `hypergraphs/` directory, create and start a virtualenv for the project 
```shell
>>> virtualenv venv
>>> source venv/bin/activate
>>> pip install -e . 
```
To use a virtualenv within jupyter notebook. Run the following command and switch to the "venv" kernel in the notebook. 
```shell 
>>> python -m ipykernel install --user --name=venv
```

### Example Code 
**Bounded-Variance Matching**  
Writes each matching to a file and writes the results (stats) of 20 matchings per gamma to a file 
```python 
from hypergraphs import hypergraphs
from hypergraphs.hypergraphs import Hypergraph
import numpy as np
import os

results_direc = 'results'
edge_list = hypergraphs.read_pickle('dblp_graph.pk')
g = Hypergraph(edges_list, prob_key='probability',weight_key='citations')
gamma_vals = np.linspace(0,2,9).tolist() + np.linspace(3,5,3).tolist()
intervals = 20

for gamma in gamma_vals:
    threshold_vals = g.gen_betas(intervals, gamma=gamma)
    gamma_results = []
    for idx, threshold in enumerate(threshold_vals):
        matching, stats =  g.bounded_var_matching(gamma, threshold)
        gamma_results.append(stats)
        f = results + '/approx_match_{}_{}.pkl'.format(gamma, idx)
        hypergraphs.writePickle(matching, f)
    f = subdirec + results + '/approx_calc_{}.pkl'.format(gamma)
    hypergraphs.writePickle(gamma_results, f)
```

**Generate Edge Probabilities Provided a Graph**
```python 
from hypergraphs import hypergraphs
from hypergraphs import graph_generator 
import numpy as np 
import pickle 

''' Weighted Graph '''
prob_types = ['UAR', 'PL']
vertices = [1000, 2000] # two graphs 
gamma_vals = np.linspace(0,2,9).tolist() + np.linspace(3,5,3).tolist() 

for p in prob_types:
    edges_list = hypergraphs.read_pickle('weighted_graph.pkl')
	graph = graph_generator.gen_attributes(edge_list, prob=p)
	# Run bounded variance matching like above on the graph 
```

**Generate a Synthetic Graph and Edge Weights and Probabilities**
```python 
from hypergraphs import graph_generator
import numpy as np 

graph_types = ['ER', 'BA']
prob_types = ['UAR', 'PL']
weight_types = ['UAR', 'PL']
vertices = [1000, 2000]
gamma_vals = np.linspace(0,2,9).tolist() + np.linspace(3,5,3).tolist() 

for g in graph_types:
    for w in weight_types:
        for p in prob_types:
            graphs = graph_generator.gen_graph(vertices, p=0.1, graph=g, prob=p, weight=w)
            for graph in graphs:
                # Run bounded variance matching like above for each graph
```
