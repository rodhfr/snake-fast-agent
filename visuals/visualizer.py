import sys
import os

# Adiciona a raiz do projeto ao path para os imports funcionarem
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import time
import pygame
import matplotlib
matplotlib.use("Agg")  # backend sem janela — funciona em qualquer ambiente
import matplotlib.pyplot as plt


from game.snake_game import SnakeEnv, SCREEN_SIZE, CELL_SIZE
from agent.agent import DQNAgent


# ─── Cores da interface ────────────────────────────────────────────────────

COR_FUNDO        = (15,  15,  15)
COR_COBRA_CABECA = (0,  220, 120)
COR_COBRA_CORPO  = (0,  160,  80)
COR_COMIDA       = (230,  60,  60)
COR_HUD          = (30,  30,  30)
COR_TEXTO        = (180, 180, 180)
COR_SCORE        = (0,  220, 120)


def load_scores(path="results/scores.csv"):
    """Lê o histórico de scores salvo pelo treino."""
    scores = []

    if not os.path.exists(path):
        return scores

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(int(row["score"]))

    return scores


def plot_scores(scores, save_path="results/graficos/evolucao_score.png"):
    """
    Gera e salva o gráfico de evolução dos scores ao longo do treino.
    Inclui a linha de score por episódio e a média móvel dos últimos 100.
    """
    if not scores:
        print("Nenhum score para plotar.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Linha de score por episódio (fina e transparente)
    ax.plot(
        scores,
        color="#5DCAA5",
        alpha=0.4,
        linewidth=1,
        label="Score por episódio",
    )

    # Média móvel dos últimos 100 episódios (linha grossa)
    if len(scores) >= 10:
        media_movel = [
            sum(scores[max(0, i - 100) : i + 1]) / min(i + 1, 100)
            for i in range(len(scores))
        ]
        ax.plot(
            media_movel,
            color="#085041",
            linewidth=2.5,
            label="Média últimos 100 ep.",
        )

    # Linha do melhor score
    ax.axhline(
        y=max(scores),
        color="#D85A30",
        linewidth=1,
        linestyle="--",
        alpha=0.6,
        label=f"Melhor score: {max(scores)}",
    )

    ax.set_xlabel("Episódio", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Evolução do Score — Snake com Deep Q-Learning", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Gráfico salvo em: {save_path}")


def run_demo(
    model_path="models/model_final.pth",
    episodes=5,
    speed=15,
):
    """
    Roda a demo visual com o agente treinado jogando ao vivo.

    Parâmetros:
        model_path → caminho do arquivo .pth com os pesos treinados
        episodes   → quantas partidas mostrar na demo
        speed      → velocidade do jogo (frames por segundo)
    """

    # Verifica se o modelo existe
    if not os.path.exists(model_path):
        print(f"\nModelo não encontrado: {model_path}")
        print("Treine o agente primeiro com: python training/train.py")
        return

    # Inicializa o pygame
    pygame.init()

    # Tela com espaço extra embaixo para o HUD
    HUD_HEIGHT = 70
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE + HUD_HEIGHT))
    pygame.display.set_caption("Snake RL — Demo do Agente Treinado")
    clock  = pygame.time.Clock()

    font_hud   = pygame.font.SysFont("monospace", 16)
    font_score = pygame.font.SysFont("monospace", 22, bold=True)

    # Inicializa o ambiente e o agente
    env   = SnakeEnv()
    agent = DQNAgent()
    agent.load(model_path)
    agent.epsilon = 0.0  # sem aleatoriedade na demo — só a rede decide

    total_score = 0

    print(f"\nIniciando demo com {episodes} episódio(s)...")
    print(f"{'Ep':>4} | {'Score':>6}")
    print("-" * 16)

    for ep in range(1, episodes + 1):

        game_state = env.reset()
        state      = agent.get_state(game_state)
        score      = 0
        done       = False

        while not done:

            # ── Eventos do pygame ─────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

            # ── Desenho do jogo ───────────────────────────────────────────

            # Fundo do jogo
            screen.fill(COR_FUNDO)

            # Corpo da cobra
            for i, segment in enumerate(game_state["snake"]):
                cor = COR_COBRA_CABECA if i == 0 else COR_COBRA_CORPO
                pygame.draw.rect(
                    screen, cor,
                    (segment[0], segment[1], CELL_SIZE - 1, CELL_SIZE - 1)
                )

            # Comida
            pygame.draw.rect(
                screen, COR_COMIDA,
                (*game_state["food"], CELL_SIZE - 1, CELL_SIZE - 1)
            )

            # ── HUD (barra inferior) ──────────────────────────────────────
            pygame.draw.rect(
                screen, COR_HUD,
                (0, SCREEN_SIZE, SCREEN_SIZE, HUD_HEIGHT)
            )
            pygame.draw.line(
                screen, (60, 60, 60),
                (0, SCREEN_SIZE), (SCREEN_SIZE, SCREEN_SIZE), 1
            )

            ep_surf    = font_hud.render(
                f"Episódio {ep}/{episodes}", True, COR_TEXTO
            )
            score_surf = font_score.render(
                f"Score: {score}", True, COR_SCORE
            )
            total_surf = font_hud.render(
                f"Melhor: {max(1, total_score)}", True, COR_TEXTO
            )

            screen.blit(ep_surf,    (14, SCREEN_SIZE + 10))
            screen.blit(score_surf, (14, SCREEN_SIZE + 32))
            screen.blit(total_surf, (SCREEN_SIZE - 120, SCREEN_SIZE + 10))

            pygame.display.update()
            clock.tick(speed)

            # ── Decisão do agente ─────────────────────────────────────────
            relative_action = agent.choose_action(state)
            abs_action      = agent.get_absolute_action(relative_action, game_state)

            game_state, reward, done = env.step(abs_action)
            state = agent.get_state(game_state)

            if reward > 0:
                score += 1

        # Fim do episódio
        total_score = max(total_score, score)
        print(f"{ep:>4} | {score:>6}")
        time.sleep(0.4)

    print(f"\nMelhor score na demo: {total_score}")
    pygame.quit()


if __name__ == "__main__":
    # 1. Gera o gráfico de evolução dos scores do treino
    scores = load_scores()
    if scores:
        plot_scores(scores)
    else:
        print("Nenhum score encontrado em results/scores.csv")
        print("Execute o treino primeiro: python training/train.py")

    # 2. Roda a demo ao vivo com o agente treinado
    run_demo()
