import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.snake_game import UP, DOWN, LEFT, RIGHT, CELL_SIZE, SCREEN_SIZE


# Ordem das direções no sentido horário — usada para calcular viradas
CLOCKWISE = [RIGHT, DOWN, LEFT, UP]


def get_direction(snake):
    """Calcula a direção atual da cobra comparando a cabeça com o segundo segmento."""

    # Se a cobra tem só 1 segmento (início do jogo), assume que vai para a direita
    if len(snake) < 2:
        return RIGHT

    head_x, head_y = snake[0]
    prev_x, prev_y = snake[1]

    # No pygame, y aumenta para BAIXO, então head_y > prev_y significa que vai DOWN
    if head_x > prev_x:
        return RIGHT
    elif head_x < prev_x:
        return LEFT
    elif head_y > prev_y:
        return DOWN
    else:
        return UP


def relative_to_absolute(relative_action, current_direction):
    """
    Converte ação relativa para ação absoluta.

    Ações relativas (o que o agente decide):
        0 → continuar reto
        1 → virar à direita (sentido horário)
        2 → virar à esquerda (sentido anti-horário)

    Ações absolutas (o que o jogo entende):
        UP=0, DOWN=1, LEFT=2, RIGHT=3
    """
    idx = CLOCKWISE.index(current_direction)

    if relative_action == 0:
        # Reto: mantém a direção atual
        return current_direction
    elif relative_action == 1:
        # Virar direita: próximo no sentido horário
        return CLOCKWISE[(idx + 1) % 4]
    else:
        # Virar esquerda: anterior no sentido horário (= sentido anti-horário)
        return CLOCKWISE[(idx - 1) % 4]


def check_collision(pos, snake):
    """Verifica se uma posição causa colisão com parede ou com o corpo da cobra."""
    x, y = pos

    # Bateu na parede
    hit_wall = x < 0 or x >= SCREEN_SIZE or y < 0 or y >= SCREEN_SIZE

    # Bateu no próprio corpo (ignora a cabeça atual — snake[1:])
    hit_self = pos in snake[1:]

    return hit_wall or hit_self


def get_state_vector(game_state):
    """
    Converte o dicionário de estado do jogo em um vetor de 11 inteiros (0 ou 1).

    Estrutura do vetor:
        [0]  danger_straight  → tem colisão se continuar reto?
        [1]  danger_right     → tem colisão se virar à direita?
        [2]  danger_left      → tem colisão se virar à esquerda?
        [3]  dir_left         → cobra está indo para a esquerda?
        [4]  dir_right        → cobra está indo para a direita?
        [5]  dir_up           → cobra está indo para cima?
        [6]  dir_down         → cobra está indo para baixo?
        [7]  food_left        → comida está à esquerda da cabeça?
        [8]  food_right       → comida está à direita da cabeça?
        [9]  food_up          → comida está acima da cabeça?
        [10] food_down        → comida está abaixo da cabeça?

    Exemplo de saída: [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
        → sem perigo reto, perigo à direita, indo para direita, comida à direita
    """
    snake = game_state["snake"]
    food  = game_state["food"]

    head_x, head_y = snake[0]
    food_x, food_y = food

    # Direção atual da cobra
    direction = get_direction(snake)

    def next_pos_for(relative_action):
        """Calcula a próxima posição da cabeça para uma ação relativa."""
        abs_action = relative_to_absolute(relative_action, direction)

        if abs_action == UP:
            return (head_x, head_y - CELL_SIZE)
        elif abs_action == DOWN:
            return (head_x, head_y + CELL_SIZE)
        elif abs_action == LEFT:
            return (head_x - CELL_SIZE, head_y)
        else:  # RIGHT
            return (head_x + CELL_SIZE, head_y)

    # --- Perigos nas 3 direções relativas ---
    danger_straight = check_collision(next_pos_for(0), snake)
    danger_right    = check_collision(next_pos_for(1), snake)
    danger_left     = check_collision(next_pos_for(2), snake)

    # --- Direção atual (one-hot: só um deles é 1) ---
    dir_left  = direction == LEFT
    dir_right = direction == RIGHT
    dir_up    = direction == UP
    dir_down  = direction == DOWN

    # --- Posição relativa da comida ---
    # No pygame y cresce para baixo, então food_y < head_y significa que a comida está ACIMA
    food_left  = food_x < head_x
    food_right = food_x > head_x
    food_up    = food_y < head_y
    food_down  = food_y > head_y

    return [
        int(danger_straight),
        int(danger_right),
        int(danger_left),
        int(dir_left),
        int(dir_right),
        int(dir_up),
        int(dir_down),
        int(food_left),
        int(food_right),
        int(food_up),
        int(food_down),
    ]
