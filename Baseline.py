# A star algorithm has to be implemented as the basedline heuristic of the agent
#  ******************************************************************************
#  ******************************************************************************
#  ******************************************************************************
#  ******************************************************************************
import environments_fully_observable 
import environments_partially_observable
import numpy as np
from  tqdm import trange
import matplotlib.pyplot as plt
import random
import tensorflow as tf
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

import queue
import heapq

import copy
import time

def display_boards(env, n=5):
    
    fig,axs=plt.subplots(1,min(len(env.boards), n), figsize=(10,3))
    for ax, board in zip(axs, env.boards):
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.imshow(board, origin="lower")



# A* algorithm of finding the path up to the fruit

class A_star_node:
    
    def __init__(self, position, cost, heuristic, parent, body, action = None):
        self.position = position
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent
        self.body = body
        self.action = action
        

    def __lt__(self, other):
        return self.cost + self.heuristic < other.cost + other.heuristic
    
    def __getitem__(self, key):
        return getattr(self, key, None)

class Heuristic_Agent:
    
    def __init__(self, env):
        self.env = env            
        self.directions = {(1, 0): self.env.UP, (-1, 0): self.env.DOWN, (0, 1): self.env.RIGHT, (0, -1): self.env.LEFT}
    def forbidden_cells(self, body):
        walls = np.argwhere(self.env.boards[0] == self.env.WALL)
        if len(body) == 0:
            # If body is empty, return only the walls
            return np.array([tuple(wall) for wall in walls])

        forbidden = np.concatenate((np.array([tuple(wall) for wall in walls]), np.array([tuple(body_) for body_ in body])), axis=0)
            
        return forbidden

    def one_step_body(self, body, head):
        new_body = copy.deepcopy(body)
        if len(new_body) > 0:
            # add head as the first element of the body
            new_body = np.insert(new_body, 0, head[1:], axis=0)
            # remove the last element of the body as tail
            new_body = np.delete(new_body, -1, axis=0)
        return new_body

    def get_neighbors(self, node, idx):
        # Get valid neighbors of a position
        neighbors = []
        forbidden_cells = set(tuple(cell) for cell in self.forbidden_cells(node["body"]))
        for move in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor_position = (node["position"][0] + move[0], node["position"][1] + move[1])
            neighbor_body = self.one_step_body(self.env.bodies[idx], node["position"])
            neighbor_action = move

            if neighbor_position not in forbidden_cells:
                neighbors.append({"position": neighbor_position, "body": np.array(neighbor_body), "action": neighbor_action})

        return neighbors 

    def heuristic(self, position, goal, body):
        distance_to_goal = abs(position[0] - goal[0]) + abs(position[1] - goal[1])
        if tuple(position) in  [tuple(cell) for cell in body]:
            if len(body) > 0:
                distance_to_goal += (len(body) ** 2)       
        return distance_to_goal

    def a_star_search(self, start, goal, idx):
        openset = []
        closed_set = set()

        heapq.heappush(openset, A_star_node(start, 0, self.heuristic(start, goal, self.env.bodies[idx]), None, self.env.bodies[idx], None))
        while len(openset) > 0:
            current_node = heapq.heappop(openset)
            if current_node.position == goal:
                action_path = []
                while current_node.parent is not None:
                    action_path.insert(0, self.directions[current_node.action])
                    current_node = current_node.parent
                return action_path

            closed_set.add(current_node.position)
            
            for neighbor in self.get_neighbors(current_node, idx):
                if neighbor["position"] not in closed_set:
                    neighbor_cost = current_node.cost + 1
                    neighbor_body = self.one_step_body(current_node.body, current_node.position)
                    neighbor_heuristic = self.heuristic(neighbor["position"], goal, neighbor_body)
                    neighbor_node = A_star_node(neighbor["position"], neighbor_cost, neighbor_heuristic, current_node, neighbor["body"], neighbor["action"])
                    neighbor_node.parent = current_node
                    if neighbor_node not in openset:
                    
                        heapq.heappush(openset, neighbor_node)
        return []  
    

    def approaching_fruit_policy(self):
        fruits = np.argwhere(self.env.boards == self.env.FRUIT)
        heads = np.argwhere(self.env.boards == self.env.HEAD)
        selected_actions = np.zeros((self.env.n_boards, 1))

        for i in range(len(self.env.boards)):
            head = tuple(heads[i][1:])
            fruit = tuple(fruits[i][1:])
            path = self.a_star_search(head, fruit, idx=i)
            if path and len(path) >= 1:
                selected_actions[i] = path[0]
        return path

    def approaching_fruit_policy2(self, paths):
        fruits = np.argwhere(self.env.boards == self.env.FRUIT)
        heads = np.argwhere(self.env.boards == self.env.HEAD)
        
        # finding the empty arrays inside paths
        indices = [i for i, path in enumerate(paths) if len(path) == 0]
        for i in indices:
            head = tuple(heads[i][1:])
            fruit = tuple(fruits[i][1:])
            path = self.a_star_search(head, fruit, idx=i)
            # add the actions inside the path into the empty array at index i
            paths[i] = path
        return paths    

    def execute(self, iteration):
        i = 0
        while i < iteration:
            actions = np.array(self.approaching_fruit_policy())
            print(actions)
            self.env.move(actions)
        display_boards(self.env, 10)

    def execute2(self, iteration):
        i = 0
        # create an array of empty arrays with the size of the boards
        paths = [[] for _ in range(self.env.n_boards)]
        print("size of paths: ", len(paths))
        while i < iteration:
            paths = self.approaching_fruit_policy2(paths)
            # print paths that are still empty
            if len([path for path in paths if len(path) == 0]) != 0:
                print(paths)
                display_boards(self.env, 10)
            actions = np.array([path.pop(0) for path in paths]).reshape(-1,1)
            self.env.move(actions)
        display_boards(self.env, 10)