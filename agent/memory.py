import random
from collections import deque


# ReplayMemory guarda todas as experiências do agente durante o treino.
# Uma "experiência" é uma tupla: (estado, ação, recompensa, próximo_estado, morreu?)
#
# Por que guardar experiências e não treinar direto?
#   Se treinássemos só com o que acabou de acontecer, a rede "esqueceria"
#   o que aprendeu antes — problema chamado "catastrophic forgetting".
#   Guardando experiências passadas e sorteando aleatoriamente, o treino
#   fica muito mais estável e eficiente.
class ReplayMemory:

    # capacity → quantas experiências guardar no máximo.
    # Quando a memória enche, a experiência mais antiga é descartada automaticamente.
    def __init__(self, capacity=100_000):

        # deque é como uma lista, mas com tamanho máximo fixo.
        # Quando atinge o limite e adicionamos algo novo,
        # o elemento mais antigo sai automaticamente pelo outro lado.
        self.memory = deque(maxlen=capacity)

    # Guarda uma nova experiência na memória.
    # Deve ser chamado após cada passo do jogo.
    def store(self, state, action, reward, next_state, done):

        # Empacotamos tudo em uma tupla e adicionamos ao final da fila
        self.memory.append((state, action, reward, next_state, done))

    # Sorteia aleatoriamente um lote de experiências para treinar a rede.
    # A aleatoriedade é fundamental — quebra a correlação entre experiências
    # consecutivas e evita que a rede fique viciada em padrões recentes.
    def sample(self, batch_size=64):

        # random.sample sorteia batch_size itens sem repetição
        batch = random.sample(self.memory, batch_size)

        # zip(*batch) transpõe a lista de tuplas em tuplas de listas.
        # Exemplo: [(s1,a1,r1), (s2,a2,r2)] → (s1,s2), (a1,a2), (r1,r2)
        # Isso facilita a conversão para tensores no Trainer.
        states, actions, rewards, next_states, dones = zip(*batch)

        # Retornamos 5 listas separadas — uma para cada componente da experiência
        return (
            list(states),
            list(actions),
            list(rewards),
            list(next_states),
            list(dones)
        )

    # Verifica se já há experiências suficientes para sortear um lote.
    # O treino só deve começar depois disso — sortear de uma memória
    # pequena demais causaria overfitting nas primeiras experiências.
    def can_sample(self, batch_size=64):

        # Retorna True se tiver pelo menos batch_size experiências guardadas
        return len(self.memory) >= batch_size

    # Permite usar len(memory) para saber quantas experiências estão guardadas
    def __len__(self):
        return len(self.memory)