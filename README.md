# MLANE
This repository provides an implementation of MLANE: Meta-Learning based Adaptive Network Embedding

## Usage
`$ python src/main.py`

## Input
Your input graph should be an **edgelist** file and make sure it under **data** folder

## Requirements
```
networkx==2.3rc1
genism==3.5.0
torch==1.2.0
scikit-learn==0.20.2
numpy==1.16.2
```

## Baselines implementations
[DeepWalk](https://github.com/phanein/deepwalk)
[node2vec](https://github.com/aditya-grover/node2vec)
[SDNE](https://github.com/thunlp/OpenNE)
[LINE](https://github.com/tangjianpku/LINE)
[HOPE](https://github.com/ZW-ZHANG/HOPE)
[AttentionWalk](https://github.com/benedekrozemberczki/AttentionWalk)
[ProNE](https://github.com/THUDM/ProNE)
[RiWalk](https://github.com/maxuewei2/RiWalk)
[Role2Vec](https://github.com/benedekrozemberczki/role2vec)
[struc2vec](https://github.com/leoribeiro/struc2vec)
[DRNE](https://github.com/tadpole/DRNE)

## Acknowledgements
We refer to [GraphEmbedding](https://github.com/shenweichen/GraphEmbedding) while constructing code framework. Thanks to the contributors for making their codes available. 
