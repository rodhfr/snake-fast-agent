import sys
import os

# Adiciona a raiz do projeto ao path para os imports funcionarem
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import pygame

from game.snake_game import SnakeEnv, SnakeRenderer
from agent.agent import DQNAgent


# ─── Configurações do Treino ──────────────────────────────────────────────

# Quantidade total de episódios (partidas) que o agente vai jogar
EPISODES = 1_000

# Mostrar o jogo na tela durante o treino (False = treino mais rápido)
RENDER = True

# Renderizar a cada N episódios (ex: 5 = só renderiza 1 a cada 5 partidas)
RENDER_EVERY = 1

# Velocidade do jogo ao renderizar (frames por segundo)
RENDER_FPS = 30


def print_header():
    print("\n" + "=" * 60)
    print("  Snake RL — Treinamento com Deep Q-Learning")
    print("=" * 60)
    print(f"  Episódios:   {EPISODES}")
    print(f"  Renderizar:  {'Sim' if RENDER else 'Não'}")
    print("=" * 60)
    print(f"{'Ep':>6} | {'Score':>6} | {'Melhor':>6} | {'Média':>6} | {'Epsilon':>7} | {'Memória':>8}")
    print("-" * 60)


def moving_average(scores, window=100):
    """Calcula a média móvel dos últimos N scores."""
    recent = scores[-window:]
    return sum(recent) / len(recent)


def train():
    # ── Inicialização ─────────────────────────────────────────────────────

    env   = SnakeEnv()
    agent = DQNAgent()

    # Cria as pastas de saída se não existirem
    os.makedirs("models",          exist_ok=True)
    os.makedirs("results/graficos", exist_ok=True)

    # Arquivo CSV para salvar o histórico de scores
    csv_path = "results/scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episodio", "score", "media_100", "epsilon"])

    # Inicializa o pygame apenas se for renderizar
    renderer   = None
    if RENDER:
        pygame.init()
        renderer = SnakeRenderer()
        # Sobe o FPS do renderer para a velocidade de treino
        renderer.clock = pygame.time.Clock()

    scores      = []
    best_score  = 0

    print_header()

    # ── Loop principal de treino ───────────────────────────────────────────
    for episode in range(1, EPISODES + 1):

        game_state = env.reset()
        state      = agent.get_state(game_state)
        score      = 0
        done       = False
        steps      = 0

        # ── Loop de um episódio ───────────────────────────────────────────
        while not done:

            # Renderiza o jogo se configurado
            should_render = RENDER and (episode % RENDER_EVERY == 0)
            if should_render:
                renderer.render(game_state)
                renderer.clock.tick(RENDER_FPS)

            # 1. Agente escolhe ação relativa (0=reto, 1=dir, 2=esq)
            relative_action = agent.choose_action(state)

            # 2. Converte para ação absoluta que o jogo entende
            abs_action = agent.get_absolute_action(relative_action, game_state)

            # 3. Executa a ação no ambiente
            next_game_state, reward, done = env.step(abs_action)
            next_state = agent.get_state(next_game_state)

            # 4. Conta ponto se comeu a comida (reward > 0)
            if reward > 0:
                score += 1

            # 5. Guarda a experiência na memória
            agent.store(state, relative_action, reward, next_state, done)

            # 6. Treino rápido com a experiência atual (1 step)
            agent.train_short(state, relative_action, reward, next_state, done)

            # 7. Treino com lote aleatório da memória (experience replay)
            agent.train_long()

            # Avança para o próximo estado
            state      = next_state
            game_state = next_game_state
            steps     += 1

        # ── Fim do episódio ───────────────────────────────────────────────

        agent.n_games += 1
        agent.decay_epsilon()
        scores.append(score)

        # Salva o melhor modelo
        if score > best_score:
            best_score = score
            agent.save("models/model_best.pth")

        # Calcula média dos últimos 100 episódios
        media = moving_average(scores)

        # Salva no CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                score,
                round(media, 2),
                round(agent.epsilon, 4),
            ])

        # Imprime progresso no terminal
        print(
            f"{episode:>6} | "
            f"{score:>6} | "
            f"{best_score:>6} | "
            f"{media:>6.1f} | "
            f"{agent.epsilon:>7.4f} | "
            f"{len(agent.memory):>8}"
        )

    # ── Fim do treino ─────────────────────────────────────────────────────

    # Salva o modelo final
    agent.save("models/model_final.pth")

    print("\n" + "=" * 60)
    print("  Treino concluído!")
    print(f"  Melhor score:        {best_score}")
    print(f"  Média últimos 100:   {moving_average(scores):.1f}")
    print(f"  Modelo salvo em:     models/model_final.pth")
    print(f"  Scores em:           results/scores.csv")
    print("=" * 60)

    if RENDER:
        pygame.quit()


if __name__ == "__main__":
    train()
