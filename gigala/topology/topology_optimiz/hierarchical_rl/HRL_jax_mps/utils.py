import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import matplotlib.pyplot as plt

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0

    def add(self, transition):
        assert len(transition) == 7, "transition must have length = 7"

        # transiton is tuple of (state, action, reward, next_state, goal, gamma, done)
        self.buffer.append(transition)
        self.size += 1

    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size / 5)]
            self.size = len(self.buffer)

        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, goals, gamma, dones = [], [], [], [], [], [], []

        for i in indexes:
            states.append(onp.array(self.buffer[i][0], copy=False))
            actions.append(onp.array(self.buffer[i][1], copy=False))
            rewards.append(onp.array(self.buffer[i][2], copy=False))
            next_states.append(onp.array(self.buffer[i][3], copy=False))
            goals.append(onp.array(self.buffer[i][4], copy=False))
            gamma.append(onp.array(self.buffer[i][5], copy=False))
            dones.append(onp.array(self.buffer[i][6], copy=False))

        return onp.array(states), onp.array(actions), onp.array(rewards), onp.array(next_states), onp.array(goals), onp.array(
            gamma), onp.array(dones)


# Define constitutive relationship.
# Generally, JAX-FEM solves -div.(f(u_grad,alpha_1,alpha_2,...,alpha_N)) = b.
# Here, we have f(u_grad,alpha_1,alpha_2,...,alpha_N) = sigma(u_grad, theta),
# reflected by the function 'stress'. The functions 'custom_init'and 'set_params'
# override base class methods. In particular, set_params sets the design variable theta.
class Elasticity(Problem):
    def custom_init(self):
        # Override base class method.
        # Set up 'self.fe.flex_inds' so that location-specific TO can be realized.
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))

    def get_tensor_map(self):
        def stress(u_grad, theta):
            # Plane stress assumption
            # Reference: https://en.wikipedia.org/wiki/Hooke%27s_law
            Emax = 70.e3
            Emin = 1e-3 * Emax
            nu = 0.3
            penal = 3.
            E = Emin + (Emax - Emin) * theta[0] ** penal
            epsilon = 0.5 * (u_grad + u_grad.T)
            eps11 = epsilon[0, 0]
            eps22 = epsilon[1, 1]
            eps12 = epsilon[0, 1]
            sig11 = E / (1 + nu) / (1 - nu) * (eps11 + nu * eps22)
            sig22 = E / (1 + nu) / (1 - nu) * (nu * eps11 + eps22)
            sig12 = E / (1 + nu) * eps12
            sigma = np.array([[sig11, sig12], [sig12, sig22]])
            return sigma

        return stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 100.])

        return [surface_map]

    def set_params(self, params):
        # Override base class method.
        full_params = np.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]

    def compute_compliance(self, sol):
        # Surface integral
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec) * # (num_selected_faces, num_face_quads, num_nodes, 1)
        u_face = sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :] * self.fe.face_shape_vals[boundary_inds[:, 1]][
                                                                          :, :, :, None]
        u_face = np.sum(u_face, axis=2)  # (num_selected_faces, num_face_quads, vec)
        # (num_cells, num_faces, num_face_quads, dim) -> (num_selected_faces, num_face_quads, dim)

        # subset_quad_points = self.get_physical_surface_quad_points(boundary_inds)

        subset_quad_points = self.physical_surface_quad_points[0]

        neumann_fn = self.get_surface_maps()[0]
        traction = -jax.vmap(jax.vmap(neumann_fn))(u_face,
                                                   subset_quad_points)  # (num_selected_faces, num_face_quads, vec)
        val = np.sum(traction * u_face * nanson_scale[:, :, None])
        return val


# Specify mesh-related information. We use first-order quadrilateral element.
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 4., 4.                                                     # dims are specific to a problem!!!
meshio_mesh = rectangle_mesh(Nx=4, Ny=4, domain_x=Lx, domain_y=Ly)  # dims are specific to a problem!!!
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


# Define boundary conditions and values.
def fixed_location(point):
    return np.isclose(point.val[0], 0., atol=1e-5)


def load_location(point):
    return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], 0., atol=0.1 * Ly + 1e-5))


def dirichlet_val(point):
    return 0.


dirichlet_bc_info = [[fixed_location] * 2, [0, 1], [dirichlet_val] * 2]

location_fns = [load_location]

# Define forward problem.
problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info,
                     location_fns=location_fns)


# Apply the automatic differentiation wrapper.
# This is a critical step that makes the problem solver differentiable.
fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})

vf = 0.5

# Define the objective function 'J_total(theta)'.
# In the following, 'sol = fwd_pred(params)' basically says U = U(theta).
def J_total(params):
    # J(u(theta), theta)
    sol_list = fwd_pred(params)
    compliance = problem.compute_compliance(sol_list[0])
    return compliance


# Prepare J_total and dJ/d(theta) that are required by the MMA optimizer.
def objectiveHandle(rho):
    # MMA solver requires (J, dJ) as inputs
    # J has shape ()
    # dJ has shape (...) = rho.shape
    J, dJ = jax.value_and_grad(J_total)(rho)
    return J, dJ


# Prepare g and dg/d(theta) that are required by the MMA optimizer.
def consHandle(rho, epoch):
    # MMA solver requires (c, dc) as inputs
    # c should have shape (numConstraints,)
    # dc should have shape (numConstraints, ...)
    def computeGlobalVolumeConstraint(rho):
        g = np.mean(rho)/vf - 1.
        return g
    c, gradc = jax.value_and_grad(computeGlobalVolumeConstraint)(rho)
    c, gradc = c.reshape((1,)), gradc[None, ...]
    return c, gradc


def fast_stopt(args, x):
    value = J_total(x)
    const = onp.mean(x)
    return value, const

