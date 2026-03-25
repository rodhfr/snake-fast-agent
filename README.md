# snake-fast-agent

Automatizando o jogo Snake com alguns modelos de IA.

## Authors

Rodolfo Franca, Vinicius Mangueira, Irlan Miguel, Kerlon

## snake.py

Implementa o jogo Snake e a API de RL.

Inclui:

- SnakeEnv (reset, step)
- SnakeRenderer (pygame)
- HumanController (teclado)
- RandomController (exemplo de aleatório)

Executar como humano:
python snake.py --human

Executar com agente aleatório:
python snake.py --random

## API do Ambiente (`SnakeEnv`)

O arquivo `snake_game.py` implementa um ambiente básico de Reinforcement Learning
para o jogo Snake.

### reset()

Reinicia o ambiente para um novo episódio.

Ações realizadas:

- reinicializa a cobra na posição inicial
- gera uma nova comida
- limpa o estado de término do episódio

Retorno:

- um dicionário representando o estado inicial do ambiente

---

### step(action)

Executa um passo do ambiente a partir de uma ação discreta
(`UP`, `DOWN`, `LEFT`, `RIGHT`).

Ações realizadas:

- atualiza a posição da cobra
- verifica colisões com parede ou corpo
- verifica se a comida foi consumida

Retorno atual:

- `state`: estado atualizado do ambiente
- `reward`: valor escalar simples (implementação provisória)
- `done`: indicador único de término do episódio

---

### Limitações atuais da API

O método `step(action)` **não implementa completamente a assinatura da Gym API**.
A Gym API é um padrão de interface para ambientes de Reinforcement Learning.

Ainda falta:

- separar o indicador de término em `terminated` e `truncated`
- retornar um dicionário `info`
- tornar a função de recompensa configurável ou externa

A definição final da lógica de recompensa e dos critérios completos de término
do episódio é responsabilidade de quem implementar os agentes ou módulos de aprendizado.

## agent.py

Contém o esqueleto de um agente de Reinforcement Learning.
Aqui ta um exemplo como pode usar esse snake_game.py em um agente.

## Dependências

```bash
pip install pygame
```

## Objetivo

Usar o jogo Snake como ambiente para desenvolver e testar agentes de IA com Reinforcement Learning.

## 🛠️ TODO — Estado Atual do Projeto

### ✅ Concluído — Pessoa 1 (Ambiente / Jogo)

- [x] Implementação completa do jogo Snake
- [x] Grid, movimento da cobra e crescimento
- [x] Geração aleatória de comida
- [x] Detecção de colisão (parede e corpo)
- [x] Implementação de `reset()`
- [x] Implementação de `step(action)`
- [x] Ambiente funcional e independente de IA
- [x] Renderização com Pygame
- [x] Controle humano (teclado)
- [x] Controlador aleatório para testes
- [x] Ambiente importável e utilizável por agentes externos

---

### 🚧 A Fazer — Pessoa 2 (Estados, Recompensas e API de RL)

#### API do Ambiente (Responsabilidade de RL)

- [ ] Atualizar a assinatura de `step(action)` para o padrão Gym:
  - retornar `terminated` (fim por falha/sucesso)
  - retornar `truncated` (fim por limite externo, ex: tempo)
  - retornar o dicionário `info`
- [ ] Definir claramente o que caracteriza:
  - término natural do episódio
  - truncamento do episódio
- [ ] Manter compatibilidade com o ambiente já existente

#### Estados

- [ ] Definir a representação de estado observável pelo agente
- [ ] Implementar função de extração de estado a partir do jogo
- [ ] Validar consistência dos estados em diferentes situações

#### Recompensas

- [ ] Definir política final de recompensas
- [ ] Integrar recompensas ao loop de interação agente–ambiente

---

### 🚧 A Fazer — Próximas Etapas

#### Agente (Pessoa 4)

- [ ] Implementar política de decisão
- [ ] Implementar aprendizado
- [ ] Estratégia de exploração (epsilon-greedy)

#### Modelo (Pessoa 3)

- [ ] Implementar modelo de aprendizado
- [ ] Implementar replay memory
- [ ] Implementar rotina de treino

#### Visualização (Pessoa 5)

- [ ] Melhorar interface visual
- [ ] Visualizar métricas de treinamento
- [ ] Preparar modo demo
