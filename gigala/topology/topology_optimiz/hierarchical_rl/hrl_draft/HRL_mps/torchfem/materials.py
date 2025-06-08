from abc import ABC, abstractmethod
from math import sqrt
from typing import Callable

import torch
from torch import Tensor


class Material(ABC):
    """Base class for material models."""

    @abstractmethod
    def __init__(self):
        self.n_state: int
        self.is_vectorized: bool
        self.C: Tensor
        pass

    @abstractmethod
    def vectorize(self, n_elem: int):
        """Create a vectorized copy of the material for `n_elm` elements."""
        pass

    @abstractmethod
    def step(self, depsilon: Tensor, epsilon: Tensor, sigma: Tensor, state: Tensor):
        """Perform a strain increment."""
        pass

    @abstractmethod
    def rotate(self, R):
        """Rotate the material with rotation matrix R."""
        pass


class IsotropicElasticity3D(Material):
    def __init__(
        self,
        E: "float | Tensor",
        nu: "float | Tenso",
        eps0: "float | Tenso" = 0.0,
    ):
        # Convert float inputs to tensors
        if isinstance(E, float):
            E = torch.tensor(E)
        if isinstance(nu, float):
            nu = torch.tensor(nu)
        if isinstance(eps0, float):
            eps0 = torch.tensor(eps0)

        # Store material properties
        self.E = E
        self.nu = nu
        self.eps0 = eps0

        # There are no internal variables
        self.n_state = 0

        # Check if the material is vectorized
        if E.dim() > 0:
            self.is_vectorized = True
        else:
            self.is_vectorized = False

        # Lame parameters
        self.lbd = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.G = self.E / (2.0 * (1.0 + self.nu))

        # Stiffness tensor
        z = torch.zeros_like(self.E)
        diag = self.lbd + 2.0 * self.G
        self.C = torch.stack(
            [
                torch.stack([diag, self.lbd, self.lbd, z, z, z], dim=-1),
                torch.stack([self.lbd, diag, self.lbd, z, z, z], dim=-1),
                torch.stack([self.lbd, self.lbd, diag, z, z, z], dim=-1),
                torch.stack([z, z, z, self.G, z, z], dim=-1),
                torch.stack([z, z, z, z, self.G, z], dim=-1),
                torch.stack([z, z, z, z, z, self.G], dim=-1),
            ],
            dim=-1,
        )

        # Stiffness tensor for shells
        self.Cs = torch.stack(
            [torch.stack([self.G, z], dim=-1), torch.stack([z, self.G], dim=-1)], dim=-1
        )

    def vectorize(self, n_elem: int):
        """Create a vectorized copy of the material for `n_elm` elements."""
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            eps0 = self.eps0.repeat(n_elem)
            return IsotropicElasticity3D(E, nu, eps0)

    def step(self, depsilon: Tensor, epsilon: Tensor, sigma: Tensor, state: Tensor):
        """Perform a strain increment."""
        epsilon_new = epsilon + depsilon
        sigma_new = sigma + torch.einsum("...ij,...j->...i", self.C, depsilon)
        state_new = state
        ddsdde = self.C
        return epsilon_new, sigma_new, state_new, ddsdde

    def rotate(self, R: Tensor):
        """Rotate the material with rotation matrix R."""
        print("Rotating an isotropic material has no effect.")
        return self


class IsotropicElasticityPlaneStress(IsotropicElasticity3D):
    """Isotropic 2D plane stress material."""

    def __init__(self, E: "float | Tensor", nu: "float | Tensor"):
        super().__init__(E, nu)

        # Overwrite the 3D stiffness tensor with a 2D plane stress tensor
        fac = self.E / (1.0 - self.nu**2)
        zero = torch.zeros_like(self.E)
        self.C = torch.stack(
            [
                torch.stack([fac, fac * self.nu, zero], dim=-1),
                torch.stack([fac * self.nu, fac, zero], dim=-1),
                torch.stack([zero, zero, fac * 0.5 * (1.0 - self.nu)], dim=-1),
            ],
            dim=-1,
        )

    def vectorize(self, n_elem: int):
        """Create a vectorized copy of the material for `n_elm` elements."""
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            return IsotropicElasticityPlaneStress(E, nu)
