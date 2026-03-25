import pygame
import random
import argparse

pygame.init()

# CONSTANTES
SCREEN_SIZE = 400
CELL_SIZE = 20
FPS = 10

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]


# LOGICA DA INTERNA DA COBRA
class SnakeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(200, 200)]
        self.food = self._spawn_food()
        self.done = False
        return self.get_state()

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True

        head_x, head_y = self.snake[0]

        if action == UP:
            new_head = (head_x, head_y - CELL_SIZE)
        elif action == DOWN:
            new_head = (head_x, head_y + CELL_SIZE)
        elif action == LEFT:
            new_head = (head_x - CELL_SIZE, head_y)
        elif action == RIGHT:
            new_head = (head_x + CELL_SIZE, head_y)

        if self._collision(new_head):
            self.done = True
            return self.get_state(), -1, True

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.food = self._spawn_food()
            reward = 1
        else:
            self.snake.pop()
            reward = -0.01

        return self.get_state(), reward, False

    def _collision(self, pos):
        x, y = pos
        hit_wall = x < 0 or x >= SCREEN_SIZE or y < 0 or y >= SCREEN_SIZE
        hit_self = pos in self.snake[1:]
        return hit_wall or hit_self

    def _spawn_food(self):
        return (
            random.randrange(0, SCREEN_SIZE, CELL_SIZE),
            random.randrange(0, SCREEN_SIZE, CELL_SIZE)
        )

    def get_state(self):
        return {
            "snake": list(self.snake),
            "food": self.food,
            "done": self.done
        }

# PYGAME RENDER
class SnakeRenderer:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.clock = pygame.time.Clock()

    def render(self, state):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.fill((0, 0, 0))

        for segment in state["snake"]:
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                (*segment, CELL_SIZE, CELL_SIZE)
            )

        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            (*state["food"], CELL_SIZE, CELL_SIZE)
        )

        pygame.display.update()
        self.clock.tick(FPS)


# Input do Jogador Humano
class HumanController:
    def __init__(self):
        self.action = RIGHT

    def get_action(self):
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


# Input Aleatorio exemplo
class RandomController:
    def get_action(self, state):
        return random.choice(ACTIONS)


def main(mode):
    env = SnakeEnv()
    renderer = SnakeRenderer()

    if mode == "human":
        controller = HumanController()
    else:
        controller = RandomController()

    state = env.reset()

    while True:
        if mode == "human":
            action = controller.get_action()
        else:
            action = controller.get_action(state)

        state, reward, done = env.step(action)
        renderer.render(state)

        if done:
            state = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", action="store_true", help="play as human")
    parser.add_argument("--random", action="store_true", help="random agent")

    args = parser.parse_args()

    if args.human == args.random:
        raise SystemExit("Use exactly one flag: --human or --random")

    mode = "human" if args.human else "random"
    main(mode)
