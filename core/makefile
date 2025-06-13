
all: train plot

train:
	python3 main.py --scenario=main --mode=train --solver=pretrain --n_node_min=5 --n_node_max=5

test:
	python3 main.py --scenario=main --mode=test --solver=active_search

debug:
	# python3 main.py --scenario=debug
	python3 main.py --scenario=debug --n_train_instance 100 --device cpu --n_embed 31 --n_hidden 15 --batch 7

plot:
	python3 main.py --scenario=plot_line --metric=actor_loss & \
	python3 main.py --scenario=plot_line --metric=critic_loss & \
	python3 main.py --scenario=plot_line --metric=avg_tour_length & \
	wait

prepare:
	python3 main.py --scenario=prepare
