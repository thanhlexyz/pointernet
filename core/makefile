
all: train plot

visualize:
	python3 main.py --scenario=plot_opt_tour

train:
	python3 main.py --scenario=main --mode=train --solver=pretrain --n_node=5

test:
	python3 main.py --scenario=main --mode=test --n_node=5 --solver=random
	python3 main.py --scenario=main --mode=test --n_node=5 --solver=pretrain
	python3 main.py --scenario=main --mode=test --n_node=5 --solver=active_search
	python3 main.py --scenario=main --mode=test --n_node=5 --solver=sampling
	python3 main.py --scenario=main --mode=test --n_node=5 --solver=optimal

debug:
	python3 main.py --scenario=debug

# plot:
# 	python3 main.py --scenario=plot_line --metric=actor_loss & \
# 	python3 main.py --scenario=plot_line --metric=critic_loss & \
# 	python3 main.py --scenario=plot_line --metric=avg_tour_length & \
# 	wait

prepare:
	python3 main.py --scenario=prepare --dataset=tsp --n_train_instance=1000000 --n_test_instance=1000
