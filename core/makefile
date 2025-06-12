
all: train plot

train:
	python3 main.py --scenario=main --mode=train --solver=pretrain --n_node=5
	python3 main.py --scenario=main --mode=train --solver=pretrain --n_node=10
	python3 main.py --scenario=main --mode=train --solver=pretrain --n_node=20

test:
	python3 main.py --scenario=main --mode=test --solver=active_search --n_node=5
	python3 main.py --scenario=main --mode=test --solver=active_search --n_node=10
	python3 main.py --scenario=main --mode=test --solver=sampling --n_node=5
	python3 main.py --scenario=main --mode=test --solver=sampling --n_node=10

debug:
	python3 main.py --scenario=debug

plot:
	python3 main.py --scenario=plot_line --metric=actor_loss & \
	python3 main.py --scenario=plot_line --metric=critic_loss & \
	python3 main.py --scenario=plot_line --metric=avg_tour_length & \
	wait

prepare:
	python3 main.py --scenario=prepare
