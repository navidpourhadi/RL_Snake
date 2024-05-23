    #  ----------------------------------------------------------------------------------------------
    # |                                                                                              |
    # |                            Comparison between RL agent (DDQN) and baseline                   |
    # |                                                                                              |
    #  ----------------------------------------------------------------------------------------------

import environments_fully_observable 
import environments_partially_observable
import numpy as np
from  tqdm import trange
import matplotlib.pyplot as plt
import tensorflow as tf
tf.random.set_seed(0)
import doubleDQN
import AStar_Heuristic
import matplotlib.pyplot as plt
import copy



# function to display the n boards with suptitle
def display_boards(env, n=5, suptitle=""):
    
    fig,axs=plt.subplots(1,min(len(env.boards), n), figsize=(5,5))
    fig.suptitle(suptitle)
    for ax, board in zip(axs, env.boards):
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.imshow(board, origin="lower")


if __name__ == "__main__":

    # get the arguments from the command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--board_size', type=int, default=15, help='board size')
    parser.add_argument('--num_boards', type=int, default=5, help='number of boards')
    parser.add_argument('--iteration', type=int, default=100, help='number of iterations')
    parser.add_argument('--gamma', type=float, default=0.9, help='value of gamma')

    args = parser.parse_args()


    # general variable (train and test)
    BOARD_SIZE = args.board_size
    GAMMA = args.gamma

    # testing variables
    NUM_BOARDS = args.num_boards
    ITERATIONS = args.iteration

    # create the environment
    ref_env_ = environments_fully_observable.OriginalSnakeEnvironment(NUM_BOARDS, BOARD_SIZE)
    RL_env_ = copy.deepcopy(ref_env_)
    AStar_env_ = copy.deepcopy(ref_env_)

    # create the agent
    input_shape = ref_env_.to_state().shape[1:]
    Double_DQN_agent = doubleDQN.DoubleDQNAgent(input_shape= input_shape, num_actions= 4, gamma= GAMMA)

    try:
        Double_DQN_agent.load_weights(file_prefix=str(BOARD_SIZE)+"_DDQN")
        print("weights loaded")
    except:
        print("no weights found, training from scratch")
        exit()



    # testing DDQN agent
    RL_reward = Double_DQN_agent.play(RL_env_, ITERATIONS)
    display_boards(RL_env_, 5, "DDQN Agent")

    # instantiating the Astar agent
    AStar_agent = AStar_Heuristic.Heuristic_Agent(AStar_env_)
    baseline_reward = AStar_agent.execute(ITERATIONS)
    display_boards(AStar_env_, 5, "A* Agent")

    mean_RL_reward = np.mean(RL_reward)
    mean_baseline_reward = np.mean(baseline_reward)

    print(f"\n\nDetails of the comparison\n -----------------------------------")
    print(f"number of steps: {ITERATIONS}")
    print(f"number of the boards: {NUM_BOARDS}")
    print(f"board size: {BOARD_SIZE}\n")
    print(f"Mean Reward of the Astar agent (baseline): {mean_baseline_reward}")
    print(f"Mean Reward of the Double DQN agent (RL): {mean_RL_reward}\n")

    print(f"Ratio between the mean rewards of RL agent and the baseline A* agent:")
    print(f"(RL) / (A*): {mean_RL_reward / mean_baseline_reward} \n")    


