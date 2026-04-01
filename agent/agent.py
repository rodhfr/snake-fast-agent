import random
import torch

from agent.model import QNetwork, Trainer
from agent.memory import ReplayMemory
from agent.state import get_state_vector, relative_to_absolute, get_direction


# ─── Hiperparâmetros do Agente ─────────────────────────────────────────────

# Quantas experiências sortear por passo de treino
BATCH_SIZE = 64

# Learning rate — tamanho do passo de ajuste a cada erro
LR = 0.001

# Fator de desconto — quanto valorizar recompensas futuras (0.9 = bastante)
GAMMA = 0.9

# Epsilon inicial: 100% exploração aleatória no começo
EPSILON_START = 1.0

# Epsilon mínimo: mesmo no final, 1% das ações ainda são aleatórias
EPSILON_END = 0.01

# Fator de decaimento por episódio (quanto epsilon diminui a cada partida)
EPSILON_DECAY = 0.995

# Capacidade máxima da memória de replay
MEMORY_CAPACITY = 100_000


class DQNAgent:
    """
    Agente de Deep Q-Learning para o jogo Snake.

    Responsabilidades:
        - Observar o estado do jogo (11 valores)
        - Escolher ações com epsilon-greedy
        - Armazenar experiências na memória de replay
        - Treinar a rede neural após cada passo
    """

    def __init__(self):
        # O cérebro do agente: rede neural com 11 entradas e 3 saídas
        self.model = QNetwork(input_size=11, hidden_size=256, output_size=3)

        # O professor: treina a rede usando a Equação de Bellman
        self.trainer = Trainer(self.model, lr=LR, gamma=GAMMA)

        # A memória: guarda até 100k experiências passadas
        self.memory = ReplayMemory(capacity=MEMORY_CAPACITY)

        # Epsilon controla a exploração vs explotação
        # Começa alto (muita aleatoriedade) e vai caindo com o tempo
        self.epsilon = EPSILON_START

        # Contador de episódios jogados
        self.n_games = 0

    def get_state(self, game_state):
        """
        Converte o estado do jogo (dicionário) para o vetor de 11 valores
        que a rede neural consegue processar.
        """
        return get_state_vector(game_state)

    def get_absolute_action(self, relative_action, game_state):
        """
        Traduz a ação relativa (0=reto, 1=direita, 2=esquerda)
        para a ação absoluta que o jogo entende (UP, DOWN, LEFT, RIGHT).
        """
        direction = get_direction(game_state["snake"])
        return relative_to_absolute(relative_action, direction)

    def choose_action(self, state):
        """
        Escolhe uma ação usando a estratégia epsilon-greedy.

        Com probabilidade epsilon → ação aleatória (exploração)
        Com probabilidade 1-epsilon → melhor ação da rede (explotação)

        Retorna um índice relativo: 0=reto, 1=direita, 2=esquerda
        """
        if random.random() < self.epsilon:
            # Exploração: escolhe uma das 3 ações ao acaso
            return random.randint(0, 2)
        else:
            # Explotação: consulta a rede neural
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                # A rede retorna 3 Q-valores — escolhemos o maior
                q_values = self.model(state_tensor)
            return int(torch.argmax(q_values).item())

    def store(self, state, action, reward, next_state, done):
        """Guarda uma experiência na memória de replay."""
        self.memory.store(state, action, reward, next_state, done)

    def train_long(self):
        """
        Treino com lote aleatório da memória (experience replay).
        Só executa quando há experiências suficientes (>= BATCH_SIZE).
        Retorna o valor do loss ou None se não treinou.
        """
        if not self.memory.can_sample(BATCH_SIZE):
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        return self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short(self, state, action, reward, next_state, done):
        """
        Treino rápido com a experiência mais recente (1 passo).
        Executado após cada ação para aprendizado imediato.
        """
        return self.trainer.train_step(
            [state], [action], [reward], [next_state], [done]
        )

    def decay_epsilon(self):
        """
        Reduz o epsilon após cada episódio.
        O agente vai ficando menos aleatório e mais estratégico com o tempo.
        """
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path="models/model_final.pth"):
        """Salva os pesos do modelo treinado em arquivo."""
        self.model.save(path)

    def load(self, path="models/model_final.pth"):
        """Carrega pesos de um modelo previamente treinado."""
        self.model.load(path)
