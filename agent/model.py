"""Neural network of the Snake game from the AI introduction project."""
import torch
import torch.nn as nn
import torch.optim as optim
import os

class QNetwork(nn.Module):
    """
    Rede neural que recebe o estado do jogo e retorna
    o valor estimado (Q-valor) para cada ação possível.
 
    Entrada:  11 valores booleanos (o que o agente enxerga)
    Oculta:   256 neurônios com ativação ReLU
    Saída:    3 valores — um para cada ação (reto, direita, esquerda)
    """

    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
 
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        def forward(self, x):
            """Passagem pela rede: recebe estado, retorna Q-valores.
            x pode ser um unico estado ou um lote (batch) de estados."""
            return self.network(x)
        def save

