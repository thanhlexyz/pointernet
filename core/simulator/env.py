import simulator as sim
import torch

class Env:

    def __init__(self, args):
        # save args
        self.args = args
        # create dataloader dict
        self.dataloader_dict = sim.dataset.create(args)
        self.loader = iter(self.dataloader_dict[args.mode])
        self.episode = 0

    def reset(self):
        # extract data
        args = self.args
        loader = self.loader
        self.episode += 1
        self.n_step = 1
        #
        instance = next(loader)
        self.x = instance['x'][0]
        self.y = instance['y'][0]
        assert self.y[0] == 0
        self.visited = torch.zeros(args.n_node)
        self.action = 0 # current node
        self.visited[0] = 1.0
        observation = torch.cat([self.x, self.visited[:, None]], dim=1)
        return observation

    def step(self, action):
        # check
        assert self.visited[action] == 0.0
        # compute reward
        reward = - ((self.x[action] - self.x[self.action]) ** 2).sum().item()
        # update observation
        self.visited[action] = 1.0
        self.action = action
        next_observation = torch.cat([self.x, self.visited[:, None]], dim=1)
        self.n_step += 1
        # done
        done = len(torch.where(self.visited == 0)[0]) <= 0
        if done:
            # add distance to start node
            reward -= ((self.x[self.action] - self.x[0]) ** 2).sum().item()
        # info
        info = {}
        return next_observation, reward, done, info
