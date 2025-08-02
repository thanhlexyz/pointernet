# Pointernet

Minimal implementation of [Neural Combinatorial Optimization with Reinforcement Learning](https://arxiv.org/abs/1611.09940)


## Usage

1. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
cd core
```

2. Prepare the dataset
- Generate random 2D TSP coordinates of $$N$$ nodes ($$X$$)
- Solve for optimal solution, a permutation of $$N$$ nodes ($$Y$$)
```bash
make prepare
```

3. Visualize the generated data

```sh
make visualize
```
![Optimal tour, 5 nodes TSP](figure/plot_opt_tour_tsp_5.jpg "Example of generated TSP instance with 5 nodes and its optimal tour")


4. Train Pointernet Actor & Critic models to solve 1M TSP instances
```sh
make train
```

Output data

- `../data/csv/*.csv`: log optimality gap of train/test dataset, train CrossEntropyLoss
- `../data/model/*.pkl`: best model by optimality gap of train dataset

5. Visualize the training progress

```sh
make plot
```

Output data

- `../data/figure/*.jpg`: line plot of log optimality gap of train/test dataset, train CrossEntropyLoss

Example: Convergence chart of training on 1000000 examples, testing on 1000 examples, for TSP problem with 10 nodes.

![Train Optimality Gap](static/plot_line_train_opt_gap_tsp_10.jpg)
![Test Optimality Gap](static/plot_line_test_opt_gap_tsp_10.jpg)
![Train Cross Entropy Loss](static/plot_line_train_loss_tsp_10.jpg)
