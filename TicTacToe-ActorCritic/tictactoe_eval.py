import torch
import numpy as np
from tictactoe_env import TicTacToeEnv
from tictactoe_actorcritic import Actor
import random

state_dim = 9
action_dim = 9

actor = Actor(state_dim, action_dim)
actor.load_state_dict(torch.load("tictactoe_actor.pth", map_location="cpu"))
actor.eval()

def print_board(board):
    symbols = {1: "X", -1: "O", 0: "."}
    for i in range(0, 9, 3):
        print(" ".join(symbols[x] for x in board[i:i+3]))
    print()

def agent_move(state):
    state_t = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        probs = actor(state_t).numpy()

    probs[state != 0] = 0.0

    if probs.sum() == 0:
        return np.random.choice(np.where(state == 0)[0])

    return np.argmax(probs)

def play():
    env = TicTacToeEnv()
    state, _ = env.reset()

    # Randomize who starts
    agent_player = random.choice([1, -1])
    human_player = -agent_player

    env.current_player = 1  # env always starts with +1 internally

    print("Board positions:")
    print("0 1 2\n3 4 5\n6 7 8\n")

    print(f"Agent is {'X' if agent_player == 1 else 'O'}")
    print(f"You are {'X' if human_player == 1 else 'O'}\n")

    done = False

    while not done:
        print_board(state)

        if env.current_player == agent_player:
            action = agent_move(state)
            print(f"Agent plays: {action}")
        else:
            action = int(input("Your move (0-8): "))
            if state[action] != 0:
                print("Illegal move. You lose.")
                return

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done:
            print_board(state)

            if reward == 1.0:
                if env.current_player == agent_player:
                    print("Agent wins.")
                else:
                    print("You win.")
            elif reward == -0.1:
                print("Draw.")
            else:
                print("Illegal move loss.")

            return

if __name__ == "__main__":
    play()
