import torch
import torch.nn as nn
import torch.optim as optim
import os


# QNetwork é o "cérebro" do agente — uma rede neural que olha para o estado
# do jogo e decide qual ação tem mais chance de dar uma boa recompensa.
# Ela herda de nn.Module, que é a classe base de toda rede no PyTorch.
class QNetwork(nn.Module):

    # input_size=11  → o agente enxerga 11 informações do jogo (perigos, direção, comida)
    # hidden_size=256 → camada intermediária com 256 neurônios para "raciocinar"
    # output_size=3  → 3 ações possíveis: reto, virar direita, virar esquerda
    def __init__(self, input_size=11, hidden_size=256, output_size=3):

        # Sempre chamar o __init__ do pai (nn.Module) antes de qualquer coisa
        super().__init__()

        # nn.Sequential empilha as camadas em ordem — a saída de uma vira entrada da próxima.
        # Estrutura da rede:
        #   11 entradas → 256 neurônios → ReLU → 256 neurônios → ReLU → 3 saídas
        self.network = nn.Sequential(

            # Primeira camada: conecta os 11 inputs aos 256 neurônios ocultos
            nn.Linear(input_size, hidden_size),

            # ReLU é a função de ativação: transforma valores negativos em 0.
            # Sem ela, a rede seria apenas uma equação linear e não aprenderia padrões complexos.
            nn.ReLU(),

            # Segunda camada oculta: permite que a rede aprenda padrões mais abstratos
            nn.Linear(hidden_size, hidden_size),

            # Mais uma ReLU para manter a não-linearidade entre as camadas
            nn.ReLU(),

            # Camada de saída: produz 3 valores, um para cada ação possível.
            # O valor mais alto indica a ação que a rede acha melhor no momento.
            nn.Linear(hidden_size, output_size)
        )

    # forward() é chamado automaticamente quando passamos dados pela rede.
    # Recebe o estado atual do jogo e retorna os 3 Q-valores (um por ação).
    def forward(self, x):

        # Passa x por todas as camadas do Sequential em sequência
        return self.network(x)

    # Salva os pesos treinados da rede em um arquivo .pth
    # Assim não precisamos treinar do zero toda vez
    def save(self, path="models/model_final.pth"):

        # Cria a pasta "models/" automaticamente se ela não existir
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # state_dict() retorna todos os pesos e biases da rede como um dicionário
        torch.save(self.state_dict(), path)
        print(f"Modelo salvo em: {path}")

    # Carrega pesos previamente salvos para continuar treinando ou fazer uma demo
    def load(self, path="models/model_final.pth"):

        # Lê o arquivo e coloca os pesos de volta na rede
        self.load_state_dict(torch.load(path))

        # eval() coloca a rede em modo de avaliação (desativa comportamentos
        # exclusivos do treino, como dropout). Sempre usar após carregar pesos.
        self.eval()
        print(f"Modelo carregado de: {path}")


# Trainer é o "professor" — ele pega experiências da memória,
# compara o que a rede previu com o que era ideal e ajusta os pesos.
class Trainer:

    # model → a QNetwork que vai ser treinada
    # lr    → learning rate: o tamanho do passo de ajuste a cada erro (0.001 é padrão)
    # gamma → fator de desconto: o quanto valorizar recompensas futuras (0.9 = bastante)
    def __init__(self, model, lr=0.001, gamma=0.9):
        self.model = model
        self.gamma = gamma

        # Adam é o otimizador mais usado em redes neurais.
        # Ele ajusta os pesos da rede com base nos gradientes calculados no backprop.
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # MSELoss calcula o Erro Quadrático Médio entre o previsto e o ideal.
        # Penaliza erros grandes mais do que erros pequenos — ideal para Q-valores.
        self.loss_fn = nn.MSELoss()

    # Recebe um lote de experiências e faz um passo de treinamento.
    # states, actions, rewards, next_states e dones são listas do mesmo tamanho.
    def train_step(self, states, actions, rewards, next_states, dones):

        # Convertemos as listas Python para tensores do PyTorch.
        # A rede só consegue processar tensores, não listas comuns.
        states      = torch.tensor(states,      dtype=torch.float)
        actions     = torch.tensor(actions,     dtype=torch.long)   # long porque são índices inteiros
        rewards     = torch.tensor(rewards,     dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones       = torch.tensor(dones,       dtype=torch.bool)

        # Se recebemos apenas 1 experiência (não um lote inteiro),
        # precisamos adicionar uma dimensão de batch para a rede processar.
        # Exemplo: tensor de shape (11,) vira (1, 11)
        if states.dim() == 1:
            states      = states.unsqueeze(0)
            actions     = actions.unsqueeze(0)
            rewards     = rewards.unsqueeze(0)
            next_states = next_states.unsqueeze(0)
            dones       = dones.unsqueeze(0)

        # Passa os estados pela rede para obter os Q-valores previstos.
        # Para cada estado, a rede retorna 3 valores (um por ação).
        q_previstos = self.model(states)

        # Clonamos os previstos para criar o "gabarito" (q_ideais).
        # Só vamos alterar o Q-valor da ação que foi realmente tomada —
        # as outras ações ficam iguais, zerando o erro delas no loss.
        q_ideais = q_previstos.clone()

        # Calculamos os Q-valores dos próximos estados sem gradiente.
        # Não queremos que o "alvo" mude enquanto treinamos — isso desestabiliza o treino.
        with torch.no_grad():
            q_proximos = self.model(next_states)

        # Para cada experiência no lote, calculamos o Q-valor ideal
        # usando a Equação de Bellman: Q = recompensa + gamma * melhor_Q_futuro
        for i in range(len(dones)):

            # Começa com a recompensa imediata (vale para todos os casos)
            q_novo = rewards[i]

            # Se o agente ainda não morreu, somamos o valor futuro descontado.
            # Se morreu (done=True), não há futuro — apenas a recompensa imediata.
            if not dones[i]:
                # gamma * max(Q_proximos) = recompensa futura esperada descontada
                q_novo = rewards[i] + self.gamma * torch.max(q_proximos[i])

            # Atualizamos só o Q-valor da ação que foi tomada nessa experiência
            q_ideais[i][actions[i]] = q_novo

        # Calcula o erro entre o que a rede previu e o que era ideal
        loss = self.loss_fn(q_previstos, q_ideais)

        # zero_grad() limpa os gradientes do passo anterior.
        # OBRIGATÓRIO antes de cada backward() — sem isso os gradientes acumulam.
        self.optimizer.zero_grad()

        # backward() calcula os gradientes de cada peso em relação ao loss
        loss.backward()

        # step() usa os gradientes para atualizar os pesos da rede
        self.optimizer.step()

        # Retorna o valor do loss para monitorar se o treino está evoluindo
        return loss.item()

