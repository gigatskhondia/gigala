import torch
from torch import Tensor


from .elements import Element
from .materials import Material
from .sparse import sparse_solve


from abc import ABC, abstractmethod
from typing import Dict, Tuple

class FEM(ABC):
    def __init__(self, nodes: Tensor, elements: Tensor, material: Material):
        """Initialize a general FEM problem."""

        # Store nodes and elements
        self.nodes = nodes
        self.elements = elements

        # Compute problem size
        self.n_dofs = torch.numel(self.nodes)
        self.n_nod = nodes.shape[0]
        self.n_dim = nodes.shape[1]
        self.n_elem = len(self.elements)

        # Initialize load variables
        self._forces = torch.zeros_like(nodes)
        self._displacements = torch.zeros_like(nodes)
        self._constraints = torch.zeros_like(nodes, dtype=torch.bool)

        # Compute mapping from local to global indices
        idx = (self.n_dim * self.elements).unsqueeze(-1) + torch.arange(self.n_dim)
        self.idx = idx.reshape(self.n_elem, -1).to(torch.int32)

        # Vectorize material
        if material.is_vectorized:
            self.material = material
        else:
            self.material = material.vectorize(self.n_elem)

        # Initialize types
        self.n_strains: int
        self.n_int: int
        self.ext_strain: Tensor
        self.etype: Element

    @property
    def forces(self) -> Tensor:
        return self._forces

    @forces.setter
    def forces(self, value: Tensor):
        if not value.shape == self.nodes.shape:
            raise ValueError("Forces must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Forces must be a floating-point tensor.")
        self._forces = value.to(self.nodes.device)

    @property
    def displacements(self) -> Tensor:
        return self._displacements

    @displacements.setter
    def displacements(self, value: Tensor):
        if not value.shape == self.nodes.shape:
            raise ValueError("Displacements must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Displacements must be a floating-point tensor.")
        self._displacements = value.to(self.nodes.device)

    @property
    def constraints(self) -> Tensor:
        return self._constraints

    @constraints.setter
    def constraints(self, value: Tensor):
        if not value.shape == self.nodes.shape:
            raise ValueError("Constraints must have the same shape as nodes.")
        if value.dtype != torch.bool:
            raise TypeError("Constraints must be a boolean tensor.")
        self._constraints = value.to(self.nodes.device)

    @abstractmethod
    def D(self, B: Tensor, nodes: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_k(self, detJ: Tensor, DCD: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_f(self, detJ: Tensor, D: Tensor, S: Tensor):
        raise NotImplementedError

    def compute_B(self) -> Tensor:
        """Null space representing rigid body modes."""
        if self.n_dim == 3:
            B = torch.zeros((self.n_dofs, 6))
            B[0::3, 0] = 1
            B[1::3, 1] = 1
            B[2::3, 2] = 1
            B[1::3, 3] = -self.nodes[:, 2]
            B[2::3, 3] = self.nodes[:, 1]
            B[0::3, 4] = self.nodes[:, 2]
            B[2::3, 4] = -self.nodes[:, 0]
            B[0::3, 5] = -self.nodes[:, 1]
            B[1::3, 5] = self.nodes[:, 0]
        else:
            B = torch.zeros((self.n_dofs, 3))
            B[0::2, 0] = 1
            B[1::2, 1] = 1
            B[1::2, 2] = -self.nodes[:, 0]
            B[0::2, 2] = self.nodes[:, 1]
        return B

    def k0(self) -> Tensor:
        """Compute element stiffness matrix for zero strain."""
        e = torch.zeros(2, self.n_int, self.n_elem, self.n_strains)
        s = torch.zeros(2, self.n_int, self.n_elem, self.n_strains)
        a = torch.zeros(2, self.n_int, self.n_elem, self.material.n_state)
        du = torch.zeros_like(self.nodes)
        dde0 = torch.zeros(self.n_elem, self.n_strains)
        self.K = torch.empty(0)
        k, _ = self.integrate_material(e, s, a, 1, du, dde0)
        return k

    def integrate_material(
            self,
            eps: Tensor,
            sig: Tensor,
            sta: Tensor,
            n: int,
            du: Tensor,
            de0: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Perform numerical integrations for element stiffness matrix."""
        # Reshape variables
        nodes = self.nodes[self.elements, :]
        du = du.reshape((-1, self.n_dim))[self.elements, :].reshape(self.n_elem, -1)

        # Initialize nodal force and stiffness
        N_nod = self.etype.nodes
        f = torch.zeros(self.n_elem, self.n_dim * N_nod)
        if self.K.numel() == 0 or not self.material.n_state == 0:
            k = torch.zeros((self.n_elem, self.n_dim * N_nod, self.n_dim * N_nod))
        else:
            k = torch.empty(0)

        for i, (w, xi) in enumerate(zip(self.etype.iweights(), self.etype.ipoints())):
            # Compute gradient operators
            b = self.etype.B(xi)
            if b.shape[0] == 1:
                dx = nodes[:, 1] - nodes[:, 0]
                J = 0.5 * torch.linalg.norm(dx, dim=1)[:, None, None]
            else:
                J = torch.einsum("jk,mkl->mjl", b, nodes)
            detJ = torch.linalg.det(J)
            if torch.any(detJ <= 0.0):
                raise Exception("Negative Jacobian. Check element numbering.")
            B = torch.einsum("jkl,lm->jkm", torch.linalg.inv(J), b)
            D = self.D(B, nodes)

            # Evaluate material response
            de = torch.einsum("jkl,jl->jk", D, du) - de0
            eps[n, i], sig[n, i], sta[n, i], ddsdde = self.material.step(
                de, eps[n - 1, i], sig[n - 1, i], sta[n - 1, i]
            )

            # Compute element internal forces
            f += w * self.compute_f(detJ, D, sig[n, i].clone())

            # Compute element stiffness matrix
            if self.K.numel() == 0 or not self.material.n_state == 0:
                DCD = torch.einsum("jkl,jlm,jkn->jmn", ddsdde, D, D)
                k += w * self.compute_k(detJ, DCD)

        return k, f

    def integrate_field(self, field: "Tensor | None" = None) -> Tensor:
        """Integrate scalar field over elements."""

        # Default field is ones to integrate volume
        if field is None:
            field = torch.ones(self.n_nod)

        # Integrate
        nodes = self.nodes[self.elements, :]
        res = torch.zeros(len(self.elements))
        for w, xi in zip(self.etype.iweights(), self.etype.ipoints()):
            N = self.etype.N(xi)
            B = self.etype.B(xi)
            J = torch.einsum("jk,mkl->mjl", B, nodes)
            detJ = torch.linalg.det(J)
            f = field[self.elements, None].squeeze() @ N
            res += w * f * detJ
        return res

    def assemble_stiffness(self, k: Tensor, con: Tensor) -> torch.sparse.Tensor:
        """Assemble global stiffness matrix."""

        # Initialize sparse matrix
        size = (self.n_dofs, self.n_dofs)
        K = torch.empty(size, layout=torch.sparse_coo)

        # Build matrix in chunks to prevent excessive memory usage
        chunks = 4
        for idx, k_chunk in zip(torch.chunk(self.idx, chunks), torch.chunk(k, chunks)):
            # Ravel indices and values
            chunk_size = idx.shape[0]
            col = idx.unsqueeze(1).expand(chunk_size, self.idx.shape[1], -1).ravel()
            row = idx.unsqueeze(-1).expand(chunk_size, -1, self.idx.shape[1]).ravel()
            indices = torch.stack([row, col], dim=0)
            values = k_chunk.ravel()

            # Eliminate and replace constrained dofs
            ci = torch.isin(idx, con)
            mask_col = ci.unsqueeze(1).expand(chunk_size, self.idx.shape[1], -1).ravel()
            mask_row = (
                ci.unsqueeze(-1).expand(chunk_size, -1, self.idx.shape[1]).ravel()
            )
            mask = ~(mask_col | mask_row)
            diag_index = torch.stack((con, con), dim=0)
            diag_value = torch.ones_like(con, dtype=k.dtype)

            # Concatenate
            indices = torch.cat((indices[:, mask], diag_index), dim=1)
            values = torch.cat((values[mask], diag_value), dim=0)

            K += torch.sparse_coo_tensor(indices, values, size=size).coalesce()

        return K.coalesce()

    def assemble_force(self, f: Tensor) -> Tensor:
        """Assemble global force vector."""

        # Initialize force vector
        F = torch.zeros((self.n_dofs))

        # Ravel indices and values
        indices = self.idx.ravel()
        values = f.ravel()

        return F.index_add_(0, indices, values)

    def solve(
            self,
            increments: Tensor = torch.tensor([0.0, 1.0]),
            max_iter: int = 10,
            rtol: torch.float32 = 1e-8,
            atol: torch.float32 = 1e-6,
            stol: torch.float32 = 1e-10,
            verbose: bool = False,
            direct: bool = None,
            device: str = None,
            return_intermediate: bool = False,
            aggregate_integration_points: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Solve the FEM problem with the Newton-Raphson method."""
        # Number of increments
        N = len(increments)

        # Null space rigid body modes for AMG preconditioner
        B = self.compute_B()

        # Indexes of constrained and unconstrained degrees of freedom
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()

        # Initialize variables to be computed
        epsilon = torch.zeros(N, self.n_int, self.n_elem, self.n_strains)
        sigma = torch.zeros(N, self.n_int, self.n_elem, self.n_strains)
        state = torch.zeros(N, self.n_int, self.n_elem, self.material.n_state)
        f = torch.zeros(N, self.n_nod, self.n_dim)
        u = torch.zeros(N, self.n_nod, self.n_dim)

        # Initialize global stiffness matrix
        self.K = torch.empty(0)

        # Initialize displacement increment
        du = torch.zeros_like(self.nodes).ravel()

        # Incremental loading
        for n in range(1, N):
            # Increment size
            inc = increments[n] - increments[n - 1]

            # Load increment
            F_ext = increments[n] * self.forces.ravel()
            DU = inc * self.displacements.clone().ravel()
            DE = inc * self.ext_strain

            # Newton-Raphson iterations
            for i in range(max_iter):
                du[con] = DU[con]

                # Element-wise integration
                k, f_int = self.integrate_material(epsilon, sigma, state, n, du, DE)

                # Assemble global stiffness matrix and internal force vector (if needed)
                if self.K.numel() == 0 or not self.material.n_state == 0:
                    self.K = self.assemble_stiffness(k, con)
                F_int = self.assemble_force(f_int)

                # Compute residual
                residual = F_int - F_ext
                residual[con] = 0.0
                res_norm = torch.linalg.norm(residual)

                # Save initial residual
                if i == 0:
                    res_norm0 = res_norm

                # Print iteration information
                if verbose:
                    print(
                        f"Increment {n} | Iteration {i + 1} | Residual: {res_norm:.5e}"
                    )

                # Check convergence
                if res_norm < rtol * res_norm0 or res_norm < atol:
                    break

                # Solve for displacement increment
                du -= sparse_solve(self.K, residual, B, stol, device, direct)

            if res_norm > rtol * res_norm0 and res_norm > atol:
                raise Exception("Newton-Raphson iteration did not converge.")

            # Update increment
            f[n] = F_int.reshape((-1, self.n_dim))
            u[n] = u[n - 1] + du.reshape((-1, self.n_dim))

        # Aggregate integration points as mean
        if aggregate_integration_points:
            epsilon = epsilon.mean(dim=1)
            sigma = sigma.mean(dim=1)
            state = state.mean(dim=1)

        if return_intermediate:
            # Return all intermediate values
            return u, f, sigma, epsilon, state
        else:
            # Return only the final values
            return u[-1], f[-1], sigma[-1], epsilon[-1], state[-1]