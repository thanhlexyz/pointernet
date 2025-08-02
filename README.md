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

To train the model
```sh
make train
```

Training data is written to
- `../data/csv/*.csv`: log optimality gap of train/test dataset, train CrossEntropyLoss
- `../data/model/*.pkl`: best model by optimality gap of train dataset

To visualize the training progress
```sh
make plot_train
```
![Actor loss](figure/plot_line_actor_loss.jpg)
![Critic loss](figure/plot_line_critic_loss.jpg)
![Avg tour length](figure/plot_line_avg_tour_length.jpg)
