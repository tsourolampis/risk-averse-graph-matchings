# Risk-Averse Matchings over Uncertain Graph Databases 

Contains an installable module, risk-averse-matching, for finding a bounded-variance or bounded-standard deviation matching on a uncertain graph or hypergraph. Also contains module for generating synthetic graphs with the following attributes:
- Erdos-Renyi, Barabasi-Albert graph models 
- Bernoulli (weight and probability parameters), Gaussian (mean and variance parameters) distributed edges
- Uniform, Gaussian, Power Law distributions to sample respective parameters w.r.t. to the edge distribution 

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

# Team 

- Charalampos E. Tsourakakis 
- Shreyas Sekar 
- Johnson Lam 
- Liu Yang 
