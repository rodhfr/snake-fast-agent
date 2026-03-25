import pygame
import random
import argparse

# Inicializa todos os módulos do pygame
pygame.init()

# =========================
# CONSTANTES DO JOGO
# =========================
SCREEN_SIZE = 400     # Tamanho da janela (400x400)
CELL_SIZE = 20        # Tamanho de cada célula do grid
FPS = 10              # Frames por segundo (velocidade do jogo)

# Ações possíveis (usadas por humano e agente)
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]


# =========================
# ENVIRONMENT (LÓGICA DO JOGO)
# =========================
class SnakeEnv:
    def __init__(self):
        # Inicializa o ambiente chamando reset
        self.reset()

    def reset(self):
        """
        Reinicia o ambiente para um novo episódio.
        Retorna o estado inicial.
        """
        self.snake = [(200, 200)]      # Corpo da cobra (cabeça no índice 0)
        self.food = self._spawn_food() # Posição da comida
        self.done = False              # Indica se o episódio terminou
        return self.get_state()

    def step(self, action):
        """
        Executa UM passo lógico do jogo.
        action: UP, DOWN, LEFT ou RIGHT

        Retorna:
        - state: estado atual do ambiente
        - reward: recompensa do passo
        - done: se o episódio terminou
        """
        if self.done:
            # Se o jogo acabou, não avança mais
            return self.get_state(), 0, True

        # Posição atual da cabeça
        head_x, head_y = self.snake[0]

        # Calcula a nova posição da cabeça com base na ação
        if action == UP:
            new_head = (head_x, head_y - CELL_SIZE)
        elif action == DOWN:
            new_head = (head_x, head_y + CELL_SIZE)
        elif action == LEFT:
            new_head = (head_x - CELL_SIZE, head_y)
        elif action == RIGHT:
            new_head = (head_x + CELL_SIZE, head_y)

        # Verifica colisão (parede ou próprio corpo)
        if self._collision(new_head):
            self.done = True
            return self.get_state(), -1, True  # Penalidade por morrer

        # Move a cobra: adiciona nova cabeça
        self.snake.insert(0, new_head)

        # Verifica se comeu a comida
        if new_head == self.food:
            self.food = self._spawn_food()
            reward = 1               # Recompensa positiva
        else:
            self.snake.pop()         # Remove a cauda
            reward = -0.01           # Penalidade pequena por passo (RL)

        return self.get_state(), reward, False

    def _collision(self, pos):
        """
        Verifica colisão com paredes ou com o próprio corpo.
        """
        x, y = pos

        # Colisão com a parede
        hit_wall = (
            x < 0 or x >= SCREEN_SIZE or
            y < 0 or y >= SCREEN_SIZE
        )

        # Colisão com o próprio corpo (exceto a cabeça atual)
        hit_self = pos in self.snake[1:]

        return hit_wall or hit_self

    def _spawn_food(self):
        """
        Gera uma posição aleatória para a comida,
        alinhada ao grid.
        """
        return (
            random.randrange(0, SCREEN_SIZE, CELL_SIZE),
            random.randrange(0, SCREEN_SIZE, CELL_SIZE)
        )

    def get_state(self):
        """
        Retorna o estado observável do ambiente.
        Esse é o estado que a IA recebe.
        """
        return {
            "snake": list(self.snake),  # Cópia do corpo da cobra
            "food": self.food,          # Posição da comida
            "done": self.done           # Estado terminal
        }


# =========================
# RENDERER (VISUAL / PYGAME)
# =========================
class SnakeRenderer:
    def __init__(self):
        # Cria a janela do pygame
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.clock = pygame.time.Clock()

    def render(self, state):
        """
        Desenha o estado atual do jogo usando pygame.
        """
        # Processa eventos de fechar janela
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # Limpa a tela (fundo preto)
        self.screen.fill((0, 0, 0))

        # Desenha a cobra
        for segment in state["snake"]:
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),                 # Verde
                (*segment, CELL_SIZE, CELL_SIZE)
            )

        # Desenha a comida
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),                     # Vermelho
            (*state["food"], CELL_SIZE, CELL_SIZE)
        )

        # Atualiza a tela
        pygame.display.update()

        # Controla o FPS
        self.clock.tick(FPS)


# =========================
# CONTROLLERS (QUEM DECIDE A AÇÃO)
# =========================
class HumanController:
    """
    Controlador humano (teclado).
    """
    def __init__(self):
        self.action = RIGHT  # Direção inicial

    def get_action(self):
        """
        Lê o teclado e retorna a ação atual.
        """
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.action = UP
                elif event.key == pygame.K_DOWN:
                    self.action = DOWN
                elif event.key == pygame.K_LEFT:
                    self.action = LEFT
                elif event.key == pygame.K_RIGHT:
                    self.action = RIGHT
            elif event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        return self.action


class RandomController:
    """
    Controlador aleatório (baseline para RL).
    """
    def get_action(self, state):
        # Escolhe uma ação aleatória
        return random.choice(ACTIONS)


# =========================
# MAIN (ESCOLHA DO MODO)
# =========================
def main(mode):
    # Cria o ambiente e o renderer
    env = SnakeEnv()
    renderer = SnakeRenderer()

    # Escolhe o controlador com base no modo
    if mode == "human":
        controller = HumanController()
    else:
        controller = RandomController()

    # Reinicia o ambiente
    state = env.reset()

    # Loop principal
    while True:
        # Decide a ação
        if mode == "human":
            action = controller.get_action()
        else:
            action = controller.get_action(state)

        # Avança o ambiente
        state, reward, done = env.step(action)

        # Renderiza o estado
        renderer.render(state)

        # Se morreu, reinicia o episódio
        if done:
            state = env.reset()


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    # Parser de argumentos de linha de comando
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", action="store_true", help="jogar como humano")
    parser.add_argument("--random", action="store_true", help="agente aleatório")

    args = parser.parse_args()

    # Garante que apenas uma flag seja usada
    if args.human == args.random:
        raise SystemExit("Use exatamente uma flag: --human ou --random")

    mode = "human" if args.human else "random"
    main(mode)
