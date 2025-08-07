import simulator as sim
import torch

class BaseSolver:

    def __init__(self, args):
        # save args
        self.args = args
        # create env
        self.env = sim.Env(args)
        # load monitor
        self.monitor = sim.Monitor(args)

    def select_action(self, observation):
        raise NotImplementedError

    def test(self):
        # extract args
        monitor, args = self.monitor, self.args
        # test each property
        for _ in range(args.n_test_episode):
            info = self.test_episode()
            monitor.step(info)
            monitor.export_csv()

    def test_episode(self):
        # extract args
        args = self.args
        env  = self.env
        # reset environment
        observation = env.reset()
        done = False
        info = {'episode': env.episode, 'value': 0}
        while not done:
            action = self.select_action(observation)
            next_observation, reward, done, _ = env.step(action)
            info['value'] += reward
            observation = next_observation
        return info
