from abc import ABC, abstractmethod
from math import sqrt
from typing import Dict, Tuple

import torch
from torch import Tensor


class Element(ABC):

    def __init__(self):
        self.nodes: int

    @abstractmethod
    def N(self, xi: Tensor) -> Tensor:
        pass

    @abstractmethod
    def B(self, xi: Tensor) -> Tensor:
        pass

    @abstractmethod
    def ipoints(self) -> Tensor:
        pass

    @abstractmethod
    def iweights(self) -> Tensor:
        pass


class Tria1(Element):
    def __init__(self):
        self.nodes = 3

    def N(self, xi: Tensor) -> Tensor:
        N_1 = 1.0 - xi[..., 0] - xi[..., 1]
        N_2 = xi[..., 0]
        N_3 = xi[..., 1]
        return torch.stack([N_1, N_2, N_3], dim=-1)

    def B(self, xi: Tensor) -> Tensor:
        if xi.dim() == 1:
            return torch.tensor([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        else:
            N = xi.shape[0]
            return torch.tensor([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]).repeat(N, 1, 1)

    def ipoints(self) -> Tensor:
        return torch.tensor([[1.0 / 3.0, 1.0 / 3.0]])

    def iweights(self) -> Tensor:
        return torch.tensor([0.5])


class Tria2(Element):
    def __init__(self):
        self.nodes = 6

    def N(self, xi: Tensor) -> Tensor:
        N_1 = (1 - xi[..., 0] - xi[..., 1]) * (1 - 2 * xi[..., 0] - 2 * xi[..., 1])
        N_2 = xi[..., 0] * (2 * xi[..., 0] - 1)
        N_3 = xi[..., 1] * (2 * xi[..., 1] - 1)
        N_4 = 4 * xi[..., 0] * (1 - xi[..., 0] - xi[..., 1])
        N_5 = 4 * xi[..., 0] * xi[..., 1]
        N_6 = 4 * xi[..., 1] * (1 - xi[..., 0] - xi[..., 1])
        return torch.stack([N_1, N_2, N_3, N_4, N_5, N_6], dim=-1)

    def B(self, xi: Tensor) -> Tensor:
        zeros = torch.zeros_like(xi[..., 0])
        return torch.stack(
            [
                torch.stack(
                    [
                        4 * xi[..., 0] + 4 * xi[..., 1] - 3,
                        4 * xi[..., 0] - 1,
                        zeros,
                        -4 * (2 * xi[..., 0] + xi[..., 1] - 1),
                        4 * xi[..., 1],
                        -4 * xi[..., 1],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        4 * xi[..., 0] + 4 * xi[..., 1] - 3,
                        zeros,
                        4 * xi[..., 1] - 1,
                        -4 * xi[..., 0],
                        4 * xi[..., 0],
                        -4 * (xi[..., 0] + 2 * xi[..., 1] - 1),
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )

    def ipoints(self) -> Tensor:
        return torch.tensor([[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]])

    def iweights(self) -> Tensor:
        return torch.tensor([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])


class Quad1(Element):
    def __init__(self):
        self.nodes = 4

    def N(self, xi: Tensor) -> Tensor:
        N_1 = (1.0 - xi[..., 0]) * (1.0 - xi[..., 1])
        N_2 = (1.0 + xi[..., 0]) * (1.0 - xi[..., 1])
        N_3 = (1.0 + xi[..., 0]) * (1.0 + xi[..., 1])
        N_4 = (1.0 - xi[..., 0]) * (1.0 + xi[..., 1])
        return 0.25 * torch.stack([N_1, N_2, N_3, N_4], dim=-1)

    def B(self, xi: Tensor) -> Tensor:
        return 0.25 * torch.stack(
            [
                torch.stack(
                    [
                        -(1 - xi[..., 1]),
                        (1 - xi[..., 1]),
                        (1.0 + xi[..., 1]),
                        -(1.0 + xi[..., 1]),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        -(1 - xi[..., 0]),
                        -(1 + xi[..., 0]),
                        (1.0 + xi[..., 0]),
                        (1.0 - xi[..., 0]),
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )

    def ipoints(self) -> Tensor:
        return torch.tensor(
            [
                [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0)]
                for xi_2 in [-1, 1]
                for xi_1 in [-1, 1]
            ]
        )

    def iweights(self) -> Tensor:
        return torch.tensor([1, 1, 1, 1])


class Quad2(Element):
    def __init__(self):
        self.nodes = 8

    def N(self, xi: Tensor) -> Tensor:
        N_1 = -(1 - xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 0] + xi[..., 1])
        N_2 = -(1 + xi[..., 0]) * (1 - xi[..., 1]) * (1 - xi[..., 0] + xi[..., 1])
        N_3 = -(1 + xi[..., 0]) * (1 + xi[..., 1]) * (1 - xi[..., 0] - xi[..., 1])
        N_4 = -(1 - xi[..., 0]) * (1 + xi[..., 1]) * (1 + xi[..., 0] - xi[..., 1])
        N_5 = 2 * (1 - xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 0])
        N_6 = 2 * (1 + xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 1])
        N_7 = 2 * (1 - xi[..., 0]) * (1 + xi[..., 1]) * (1 + xi[..., 0])
        N_8 = 2 * (1 - xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 1])
        return 0.25 * torch.stack([N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8], dim=-1)

    def B(self, xi: Tensor) -> Tensor:
        return 0.25 * torch.stack(
            [
                torch.stack(
                    [
                        -(xi[..., 1] - 1) * (2 * xi[..., 0] + xi[..., 1]),
                        -(xi[..., 1] - 1) * (2 * xi[..., 0] - xi[..., 1]),
                        +(xi[..., 1] + 1) * (2 * xi[..., 0] + xi[..., 1]),
                        +(xi[..., 1] + 1) * (2 * xi[..., 0] - xi[..., 1]),
                        +4 * xi[..., 0] * (xi[..., 1] - 1),
                        +2 - 2 * xi[..., 1] ** 2,
                        -4 * xi[..., 0] * (xi[..., 1] + 1),
                        -2 + 2 * xi[..., 1] ** 2,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        -(xi[..., 0] - 1) * (xi[..., 0] + 2 * xi[..., 1]),
                        -(xi[..., 0] + 1) * (xi[..., 0] - 2 * xi[..., 1]),
                        +(xi[..., 0] + 1) * (xi[..., 0] + 2 * xi[..., 1]),
                        +(xi[..., 0] - 1) * (xi[..., 0] - 2 * xi[..., 1]),
                        -2 + 2 * xi[..., 0] ** 2,
                        -4 * xi[..., 1] * (xi[..., 0] + 1),
                        +2 - 2 * xi[..., 0] ** 2,
                        +4 * (xi[..., 0] - 1) * xi[..., 1],
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )

    def ipoints(self) -> Tensor:
        return torch.tensor(
            [
                [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0)]
                for xi_2 in [-1, 1]
                for xi_1 in [-1, 1]
            ]
        )

    def iweights(self) -> Tensor:
        return torch.tensor([1, 1, 1, 1])
