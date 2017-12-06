# Risk Averse Team Formations 
Contains an installable module, risk-averse-matching, for finding a bounded-variance matching on a standard graph or hypergraph and generating synthetic graphs, edge weight, and edge probabilities. 

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

### TODO
- Update README w/ example code for generating synthetic graphs and running a matching 
