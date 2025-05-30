# Pointer Net

Reproduce this paper
> Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer networks. Advances in neural information processing systems, 28.

## How to run the code

1. Install dependencies
```sh
pip3 install -r requirements.txt
```

2. Prepare the dataset

- random 2D TSP coordinates of $$N$$ nodes ($$X$$)
- optimal solution, a permutation of $$N$$ nodes ($$Y$$)

```sh
make prepare
```
Change `args.n_instance` and `args.n_node` depending on your experimental setup.

3. Visualize the generated data

```sh
make plot
```

![plot_tsp_10.pdf](static/plot_tsp_10.pdf)


## TODO

- [x] Visualize the generated tours
- [ ] Inference (test) from random (untrained) model
