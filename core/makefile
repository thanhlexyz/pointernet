
all: train plot

train:
	python3 main.py --scenario=main

plot:
	python3 main.py --scenario=plot_line --metric=train_loss & \
	python3 main.py --scenario=plot_line --metric=train_opt_gap & \
	python3 main.py --scenario=plot_line --metric=test_opt_gap & \
	wait

prepare:
	python3 main.py --scenario=prepare
