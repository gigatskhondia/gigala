import numpy as np
import autograd.numpy as anp      
import scipy, scipy.ndimage, scipy.sparse, scipy.sparse.linalg  
import torch

device = torch.device("mps:0" if torch.mps.is_available() else "cpu")
# device = torch.device("cpu" if torch.mps.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        assert len(transition) == 7, "transition must have length = 7"
        
        # transiton is tuple of (state, action, reward, next_state, goal, gamma, done)
        self.buffer.append(transition)
        self.size +=1
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)
        
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, goals, gamma, dones = [], [], [], [], [], [], []
        
        for i in indexes:
            states.append(np.array(self.buffer[i][0], copy=False))
            actions.append(np.array(self.buffer[i][1], copy=False))
            rewards.append(np.array(self.buffer[i][2], copy=False))
            next_states.append(np.array(self.buffer[i][3], copy=False))
            goals.append(np.array(self.buffer[i][4], copy=False))
            gamma.append(np.array(self.buffer[i][5], copy=False))
            dones.append(np.array(self.buffer[i][6], copy=False))
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(goals),  np.array(gamma), np.array(dones)
    

# TO and FEM model
#
# class ObjectView(object):
#     def __init__(self, d): self.__dict__ = d
#
# def get_args(normals, forces, density=1e-4):  # Manage the problem setup parameters
#     width = normals.shape[0] - 1
#     height = normals.shape[1] - 1
#     fixdofs = np.flatnonzero(normals.ravel())
#     alldofs = np.arange(2 * (width + 1) * (height + 1))
#     freedofs = np.sort(list(set(alldofs) - set(fixdofs)))
#     params = {
#       # material properties
#       'young': 1, 'young_min': 1e-9, 'poisson': 0.3, 'g': 0,
#       # constraints
#       'density': density, 'xmin': 0.001, 'xmax': 1.0,
#       # input parameters
#       'nelx': width, 'nely': height, 'mask': 1, 'penal': 3.0, 'filter_width': 1,
#       'freedofs': freedofs, 'fixdofs': fixdofs, 'forces': forces.ravel(),
#       # optimization parameters
#       'opt_steps': 80, 'print_every': 10}
#     return ObjectView(params)
#
#
# def mbb_beam(width=5, height=5, density=1e-4, y=1, x=0, rd=-1):  # textbook beam example                     # new line
#     normals = np.zeros((width + 1, height + 1, 2))
#     normals[0, 0, x] = 1
#     normals[0, 0, y] = 1
#     normals[0, -1, x] = 1
#     normals[0, -1, y] = 1
#     forces = np.zeros((width + 1, height + 1, 2))
#     forces[-1, rd, y] = -1
#     # print(normals,forces,density)
#     # return torch.Tensor.cpu(torch.from_numpy(normals.astype(np.float32))).to(device), torch.Tensor.cpu(torch.from_numpy(forces.astype(np.float32))).to(device), torch.Tensor.cpu(torch.from_numpy(np.array(density).astype(np.float32))).to(device)
#     return normals,forces,density
# def young_modulus(x, e_0, e_min, p=3):
#     return e_min + x ** p * (e_0 - e_min)
#
# def physical_density(x, args, volume_contraint=False, use_filter=True):
#     x = args.mask * x.reshape(args.nely, args.nelx)  # reshape from 1D to 2D
#     return gaussian_filter(x, args.filter_width) if use_filter else x  # maybe filter
#
# def mean_density(x, args, volume_contraint=False, use_filter=True):
#     # return torch.from_numpy((anp.mean(physical_density(x, args, volume_contraint, use_filter)) / anp.mean(args.mask)).astype(np.float32)).to(device)
#     return anp.mean(physical_density(x, args, volume_contraint, use_filter)) / anp.mean(args.mask)
# def objective(x, args, volume_contraint=False, use_filter=True):
#     kwargs = dict(penal=args.penal, e_min=args.young_min, e_0=args.young)
#     x_phys = physical_density(x, args, volume_contraint=volume_contraint, use_filter=use_filter)
#     ke     = get_stiffness_matrix(args.young, args.poisson)  # stiffness matrix
#     u      = displace(x_phys, ke, args.forces, args.freedofs, args.fixdofs, **kwargs)
#     c      = compliance(x_phys, u, ke, **kwargs)
#     # return torch.from_numpy(c.astype(np.float32)).to(device)
#     return c
# def gaussian_filter(x, width): # 2D gaussian blur/filter
#     # return torch.from_numpy(scipy.ndimage.gaussian_filter(x, width, mode='reflect').astype(np.float32)).to(device)
#     return scipy.ndimage.gaussian_filter(x, width, mode='reflect')
# def _gaussian_filter_vjp(ans, x, width): # gives the gradient of orig. function w.r.t. x
#     del ans, x  # unused
#     # return lambda g: torch.from_numpy(gaussian_filter(g, width).astype(np.float32)).to(device)
#     return lambda g: gaussian_filter(g, width)
#
# def compliance(x_phys, u, ke, *, penal=3, e_min=1e-9, e_0=1):
#     nely, nelx = x_phys.shape
#     ely, elx = anp.meshgrid(range(nely), range(nelx))  # x, y coords for the index map
#
#     n1 = (nely+1)*(elx+0) + (ely+0)  # nodes
#     n2 = (nely+1)*(elx+1) + (ely+0)
#     n3 = (nely+1)*(elx+1) + (ely+1)
#     n4 = (nely+1)*(elx+0) + (ely+1)
#     all_ixs = anp.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])
#     u_selected = u[all_ixs]  # select from u matrix
#
#     ke_u = anp.einsum('ij,jkl->ikl', ke, u_selected)  # compute x^penal * U.T @ ke @ U
#     ce = anp.einsum('ijk,ijk->jk', u_selected, ke_u)
#     C = young_modulus(x_phys, e_0, e_min, p=penal) * ce.T
#     # return torch.from_numpy(anp.sum(C).astype(np.float32)).to(device)
#     return anp.sum(C)
# def get_stiffness_matrix(e, nu):  # e=young's modulus, nu=poisson coefficient
#     k = anp.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
#                 -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
#     return (e/(1-nu**2)*anp.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
#                                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
#                                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
#                                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
#                                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
#                                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
#                                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
#                                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]]))
#
# def get_k(stiffness, ke):
#     # Constructs sparse stiffness matrix k (used in the displace fn)
#     # First, get position of the nodes of each element in the stiffness matrix
#     nely, nelx = stiffness.shape
#     ely, elx = anp.meshgrid(range(nely), range(nelx))  # x, y coords
#     ely, elx = ely.reshape(-1, 1), elx.reshape(-1, 1)
#
#     n1 = (nely+1)*(elx+0) + (ely+0)
#     n2 = (nely+1)*(elx+1) + (ely+0)
#     n3 = (nely+1)*(elx+1) + (ely+1)
#     n4 = (nely+1)*(elx+0) + (ely+1)
#     edof = anp.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])
#     edof = edof.T[0]
#     x_list = anp.repeat(edof, 8)  # flat list pointer of each node in an element
#     y_list = anp.tile(edof, 8).flatten()  # flat list pointer of each node in elem
#
#     # make the global stiffness matrix K
#     kd = stiffness.T.reshape(nelx*nely, 1, 1)
#     value_list = (kd * anp.tile(ke, kd.shape)).flatten()
#     return value_list, y_list, x_list
#
# def displace(x_phys, ke, forces, freedofs, fixdofs, *, penal=3, e_min=1e-9, e_0=1):
#     # Displaces the load x using finite element techniques (solve_coo=most of runtime)
#     stiffness = young_modulus(x_phys, e_0, e_min, p=penal)
#     k_entries, k_ylist, k_xlist = get_k(stiffness, ke)
#
#     index_map, keep, indices = _get_dof_indices(freedofs, fixdofs, k_ylist, k_xlist)
#
#     u_nonzero = solve_coo(k_entries[keep], indices, forces[freedofs], sym_pos=True)
#     u_values = anp.concatenate([u_nonzero, anp.zeros(len(fixdofs))])
#     # return torch.from_numpy(u_values[index_map].astype(np.float32)).to(device)
#     return u_values[index_map]
#
# def _get_dof_indices(freedofs, fixdofs, k_xlist, k_ylist):
#     index_map = inverse_permutation(anp.concatenate([freedofs, fixdofs]))
#     keep = anp.isin(k_xlist, freedofs) & anp.isin(k_ylist, freedofs)
#     # Now we index an indexing array that is being indexed by the indices of k
#     i = index_map[k_ylist][keep]
#     j = index_map[k_xlist][keep]
#     return index_map, keep, anp.stack([i, j])
#
# def inverse_permutation(indices):  # reverses an index operation
#     inverse_perm = np.zeros(len(indices), dtype=anp.int64)
#     inverse_perm[indices] = np.arange(len(indices), dtype=anp.int64)
#     # return torch.from_numpy(inverse_perm.astype(np.float32)).to(device)
#     return inverse_perm
# def _get_solver(a_entries, a_indices, size, sym_pos):
#     # a is (usu.) symmetric positive; could solve 2x faster w/sksparse.cholmod.cholesky(a).solve_A
#     a = scipy.sparse.coo_matrix((a_entries, a_indices), shape=(size,)*2).tocsc()
#     return scipy.sparse.linalg.splu(a).solve
#
# def solve_coo(a_entries, a_indices, b, sym_pos=False):
#     solver = _get_solver(a_entries, a_indices, b.size, sym_pos)
#     # return torch.from_numpy(solver(b).astype(np.float32)).to(device)
#     return solver(b)
# def grad_solve_coo_entries(ans, a_entries, a_indices, b, sym_pos=False):
#     def jvp(grad_ans):
#         lambda_ = solve_coo(a_entries, a_indices if sym_pos else a_indices[::-1],
#                             grad_ans, sym_pos)
#         i, j = a_indices
#         return -lambda_[i] * ans[j]
#     # return torch.from_numpy(jvp.astype(np.float32)).to(device)
#     return jvp
# def fast_stopt(args, x):
#
#     reshape = lambda x: x.reshape(args.nely, args.nelx)
#     objective_fn = lambda x: objective(reshape(x), args)
#     # constraint = lambda params: mean_density(reshape(params), args) - args.density
#     constraint = lambda params: mean_density(reshape(params), args)
#     value = objective_fn(x)
#     const = constraint(x)
#     return torch.Tensor.cpu(torch.from_numpy(np.asarray(value).astype(np.float32)).to(device)), torch.Tensor.cpu(torch.from_numpy(np.asarray(const).astype(np.float32)).to(device))


# The 'classic' 60x20 2d mbb beam, as per Ole Sigmund's 99 line code.
config = {
    "FILT_RAD": 1.5,
    "FXTR_NODE_X": range(1, 6),
    "FXTR_NODE_Y": 36,
    "LOAD_NODE_Y": 1,
    "LOAD_VALU_Y": -1,
    "NUM_ELEM_X": 5,
    "NUM_ELEM_Y": 5,
    "NUM_ITER": 94,
    "P_FAC": 3.0,
    "VOL_FRAC": 0.5,
}


import torch

torch.set_default_dtype(torch.double)

from torchfem.planar import Planar
from torchfem.materials import IsotropicElasticityPlaneStress

# Material model (plane stress)
material = IsotropicElasticityPlaneStress(E=100.0, nu=0.3)

Nx = config["NUM_ELEM_X"]
Ny = config["NUM_ELEM_Y"]

# Create nodes
n1 = torch.linspace(0.0, Nx, Nx + 1)
n2 = torch.linspace(Ny, 0.0, Ny + 1)
n1, n2 = torch.stack(torch.meshgrid(n1, n2, indexing="ij"))
nodes = torch.stack([n1.ravel(), n2.ravel()], dim=1)

# Create elements connecting nodes
elements = []
for j in range(Ny):
    for i in range(Nx):
        n0 = j + i * (Ny + 1)
        elements.append([n0, n0 + 1, n0 + Ny + 2, n0 + Ny + 1])
elements = torch.tensor(elements)

model = Planar(nodes, elements, material)

# Load at top
model.forces[torch.tensor(config["LOAD_NODE_Y"]) - 1, 1] = config["LOAD_VALU_Y"]

# Constrained displacement at left end
model.constraints[torch.tensor(config["FXTR_NODE_X"]) - 1, 0] = True
model.constraints[torch.tensor(config["FXTR_NODE_Y"]) - 1, 1] = True


# Plot the domain
# model.plot()


# Initial, minimum, and maximum values of design variables
rho_0 = config["VOL_FRAC"] * torch.ones(len(elements), requires_grad=True)
rho_min = 0.01 * torch.ones_like(rho_0)
rho_max = torch.ones_like(rho_0)

# Volume fraction
V_0 = config["VOL_FRAC"] * Nx * Ny

# Analytical gradient of the stiffness matrix
k0 = torch.einsum("i,ijk->ijk", 1.0 / model.thickness, model.k0())

# Move limit for optimality condition algortihm
move = 0.2

# Precompute filter weights
if config["FILT_RAD"] > 0.0:
    ecenters = torch.stack([torch.mean(nodes[e], dim=0) for e in elements])
    dist = torch.cdist(ecenters, ecenters)
    H = config["FILT_RAD"] - dist
    H[dist > config["FILT_RAD"]] = 0.0



p = config["P_FAC"]

TORCH_SENS = False


def fast_stopt(dummy, rho_0):
    model.thickness = rho_0 ** p

    u_k, f_k, _, _, _ = model.solve()
    compliance = torch.inner(f_k.ravel(), u_k.ravel())
    return  compliance, torch.mean(rho_0)


# print(fast_stopt(None, rho_0))