# A* algorithm has to be implemented as the basedline heuristic of the agent
# ******************************************************************************
# ******************************************************************************
# ******************************************************************************
# ******************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import heapq
np.random.seed(0)


def display_boards(env, n=5):
    
    fig,axs=plt.subplots(1,min(len(env.boards), n), figsize=(10,3))
    for ax, board in zip(axs, env.boards):
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.imshow(board, origin="lower")


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
    
    # create an array of forbidden cells (walls and body)
    def forbidden_cells(self, body):
        walls = np.argwhere(self.env.boards[0] == self.env.WALL)
        if len(body) == 0:
            # If body is empty, return only the walls
            return np.array([tuple(wall) for wall in walls])

        forbidden = np.concatenate((np.array([tuple(wall) for wall in walls]), np.array([tuple(body_) for body_ in body])), axis=0)
            
        return forbidden


    # the body of the snake in one step ahead
    def one_step_body(self, body, head):
        return np.insert(body, 0, head[1:], axis=0)[:-1] if len(body) > 0 else body


    # get allowed neighbors of a position
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

    # heuristic function
    def heuristic(self, position, goal, body):
        distance_to_goal = abs(position[0] - goal[0]) + abs(position[1] - goal[1])
        if tuple(position) in  [tuple(cell) for cell in body]:
            if len(body) > 0:
                distance_to_goal += (len(body) ** 2)       
        return distance_to_goal


    # a* search algorithm using heapq, returns a list of actions for each board until achieving the fruit
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
        return [np.random.choice([self.env.UP, self.env.DOWN, self.env.LEFT, self.env.RIGHT])]  
    
    # approaching fruit policy using a*
    def approaching_fruit_policy(self, paths):
        fruits = np.argwhere(self.env.boards == self.env.FRUIT)
        heads = np.argwhere(self.env.boards == self.env.HEAD)
        
        # finding the empty arrays inside paths
        indices = [i for i, path in enumerate(paths) if len(path) == 0]
        for i in indices:
            head = tuple(heads[i][1:])
            fruit = tuple(fruits[i][1:])
            path = self.a_star_search(head, fruit, idx=i)
            paths[i] = path
        return paths    

    # execution of A* algorithm
    def execute(self, iteration):
        i = 0
        rewards = np.zeros(self.env.n_boards, dtype=float)
        # create an array of empty arrays with the size of the boards
        paths = [[] for _ in range(self.env.n_boards)]
        while i < iteration:
            i += 1
            paths = self.approaching_fruit_policy(paths)
            actions = np.array([path.pop(0) for path in paths]).reshape(-1,1)
            rewards = np.add(rewards, self.env.move(actions))
        display_boards(self.env, 10)
        print("rewards: ", rewards)