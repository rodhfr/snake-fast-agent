# snake-fast-agent

Automatizando o jogo Snake com alguns modelos de IA.

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

## Dependências

```bash
pip install pygame
```

## Objetivo

Usar o jogo Snake como ambiente para desenvolver e testar agentes de IA com Reinforcement Learning.
