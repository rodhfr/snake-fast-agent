from snake import SnakeEnv, ACTIONS
import random


class Agent:
    def __init__(self):
        # inicialização do modelo / Q-table / rede neural
        pass

    def get_state(self, state):
        """
        Converte o estado do ambiente (dict)
        para a representação usada pelo agente.
        """
        pass

    def choose_action(self, state):
        """
        Define a política do agente.
        """
        pass

    def learn(self, state, action, reward, next_state, done):
        """
        Atualiza o modelo do agente.
        """
        pass


def train():
    env = SnakeEnv()
    agent = Agent()

    episodes = 1000

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            s = agent.get_state(state)
            action = agent.choose_action(s)

            next_state, reward, done = env.step(action)

            s_next = agent.get_state(next_state)
            agent.learn(s, action, reward, s_next, done)

            state = next_state


if __name__ == "__main__":
    train()
