
import torch
from fenics import *
import dolfin as df
from fenics_adjoint import *
import torch_fenics
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as TorchFunction
import numpy as np
import random
import time
import fenics_adjoint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self,*args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

from mpi4py import MPI as PYMPI

commmpi = PYMPI.COMM_WORLD
rank = commmpi.Get_rank()
size = commmpi.Get_size()
root = 0
if rank == 0:
    set_log_level(20)
else:
    set_log_level(80)

parameters["form_compiler"]["quadrature_degree"] = 3


filename = "output_Multi-materials/0D35/"
os.makedirs(filename, exist_ok=True)

# -----------------------------------------
DX, DY, nuel = 2.0, 1.0, 100
volume = DX*DY
mesh = RectangleMesh(Point(0.0,0.0), Point(DX, DY), int(DX*nuel), int(DY*nuel),'crossed')
print('num cells/elements', mesh.num_cells())
print('num verticies', mesh.num_vertices())

# Function space
V = VectorFunctionSpace(mesh, 'CG', 1)
DG = FunctionSpace(mesh, 'DG', 0)
Q = FunctionSpace(mesh, "CG", 1)
# ------------------------------------------

# NN 
lag_lambda, lag_mu, lag_mu2 = 10., 40., 50.  # Lagrange penalty parameter
maxEpochs, learning_rate, nrmThreshold = 150, 0.01, 0.1  # nrmThreshold for gradient clipping

# Topopt user setting
ft = 2     # 0:net       1:net+pde        2:net+hv          3:net+pde+hv
massfrac = 0.35
useIterative = 1

# PDE setting
filterR = 7/nuel
print('filter type:', ft)
print("filter radius:", filterR)

PenaltyP = 3.
nu = 0.3
RAMP = 0
# Definition of the Lame's parameters
lmbda, mu = 1.0*nu/((1 + nu)*(1 - 2*nu)), 1.0/(2*(1 + nu))

# load setting
yload = 1		        # load to be applied on the y axis
app_pt = Point(DX, DY/2)	# application point of the load

# -----------------------------------------------------------------------------

class Timer:
    def __init__(self, name=""):
        self.name = name
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        # print(f"[{self.name}] Time elapsed: {self.interval:.4f} s")

timing_stats = {
    "Forward_net": 0.0,
    "FEA_loss": 0.0,
    "Backward": 0.0,
    "Optim_step": 0.0,
}

# convert np to torch, torch to np
def to_torch(x):
  #return torch.tensor(x).double()
  return torch.tensor(x,dtype=torch.float64)
def to_np(x):
  return x.detach().cpu().numpy()

def setDevice(overrideGPU=False):
    if torch.cuda.is_available() and not overrideGPU:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

device = setDevice(overrideGPU=False)
torch.autograd.set_detect_anomaly(True)

def set_seed(manualSeed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)

#%% Neural network
class TopNet(nn.Module):
    inputDim = 2; # x and y coordn of the point
    def __init__(self, numLayers, numNeuronsPerLyr, nelx, nely, symXAxis, symYAxis):

        # ---- import materials inf from json ----
        with open("materials.json", "r") as f:
            mat_config = json.load(f)

        self.materials = mat_config["materials"]
        self.E_list = [mat["E"] for mat in self.materials]
        self.num_materials = self.outputDim = len(self.E_list)

        self.nelx = nelx; # to impose symm, get size of domain
        self.nely = nely; 
        self.symXAxis = symXAxis; # set T/F to impose symm
        self.symYAxis = symYAxis;
        super().__init__();
        self.layers = nn.ModuleList();
        current_dim = self.inputDim;
        manualSeed = 221; # NN are seeded manually 
        set_seed(manualSeed)
        for lyr in range(numLayers): # define the layers
            l = nn.Linear(current_dim, numNeuronsPerLyr);

            nn.init.xavier_normal_(l.weight);             # "xavier"
            nn.init.zeros_(l.bias);
            self.layers.append(l);
            current_dim = numNeuronsPerLyr;
        self.layers.append(nn.Linear(current_dim, self.outputDim));
        self.bnLayer = nn.ModuleList();
        for lyr in range(numLayers): # batch norm 
            self.bnLayer.append(nn.BatchNorm1d(numNeuronsPerLyr));
    def forward(self, x, fixedIdx = None):
        m = nn.LeakyReLU();
        ctr = 0;
        if(self.symXAxis):
            xv = 0.5*self.nelx + torch.abs( x[:,0] - 0.5*self.nelx);
        else:
            xv = x[:,0];
        if(self.symYAxis):
            yv = 0.5*self.nely + torch.abs( x[:,1] - 0.5*self.nely) ;
        else:
            yv = x[:,1];

        x = torch.transpose(torch.stack((xv,yv)),0,1);
        for layer in self.layers[:-1]: # forward prop
            x = m(self.bnLayer[ctr](layer(x)));
            ctr += 1;
        out = torch.softmax(self.layers[-1](x), dim = 1).T;  # output layer
        
        print(out)
        return  out;



def compute_material_fields(gamma_list, E_list, rho_list):
    E_eff = sum(gamma**PenaltyP * Constant(E0) for gamma, E0 in zip(gamma_list, E_list))
    rho_eff = sum(gamma * Constant(rho0) for gamma, rho0 in zip(gamma_list, rho_list))
    return E_eff, rho_eff



def es(v,lmbda,mu):  # Mechanical stress
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(v.geometric_dimension())

# Solve the FE problem
class FEA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gamma_tensor, loop_iter):
        
        with open("materials.json", "r") as f:
            mat_config = json.load(f)
        materials = mat_config["materials"]
        E_list = [mat["E"] for mat in materials]
        nu_list = [mat["nu"] for mat in materials]
        rho_list = [mat["rho"] for mat in materials]
        print("E list:", E_list, '\n'
              "Rho list:", rho_list )
        
        gamma_funcs = []
        for i, gamma_torch in enumerate(gamma_tensor):
            gamma = Function(Q)
            gamma.vector()[:] = gamma_torch.detach().cpu().numpy().reshape(-1)
            gamma_funcs.append(gamma)
            
        E_eff, rho_eff = compute_material_fields(gamma_funcs, E_list, rho_list)
       
        # Boundary condition
        fix_boundary = 'near(x[0], 0) && ((x[1] >= 0.0 && x[1] <= 0.2) || (x[1] >= 0.8 && x[1] <= 1)) && on_boundary'
        fix_bc = DirichletBC(V, Constant((0, 0)), fix_boundary)
        bcs = [fix_bc]
        
        # Create trial and test functions
        u_ = TrialFunction(V)
        tu = TestFunction(V)
        f = Constant((0.0, 0.0))
        
        A = E_eff*inner(grad(tu), es(u_,lmbda, mu))*dx
        l_ = inner(tu,f)*dx
    
        elasticity_A, elasticity_b = assemble_system(A, l_, bcs)
        
        # load
        f_delta = PointSource(V.sub(1), app_pt, yload) 
        f_delta.apply(elasticity_b)

        Dis = Function(V)
        solve(elasticity_A, Dis.vector(), elasticity_b, 'mumps')
                
        # obj function
        compliance = assemble(E_eff*inner(grad(Dis), es(Dis, lmbda, mu))*dx)
        mass       = assemble(rho_eff*dx)
        global mass_diff
        mass_diff   = mass/(volume*max(rho_list))-massfrac

       
        if loop_iter == 1:
            global compliance0
            compliance0 = assemble(inner(grad(Dis), es(Dis,lmbda, mu))*dx)

        score = 0.7*compliance/compliance0 + 0.3*loop_iter/maxEpochs
        
        cost_comp = compliance/compliance0
        #cost_mass = lag_lambda*mass_diff + 0.5*lag_mu*mass_diff**2
        cost_mass = lag_mu2*mass_diff**2
        
        cost = cost_comp + cost_mass
        print("Iter:", loop_iter, "cost:", cost, "compliance:", compliance, "mass:", mass/(volume*max(rho_list)), "score:", score)

        history_file.write('%s\t %s\t %s\t %s\t %s\n' % (loop_iter, cost, compliance, mass/(volume*max(rho_list)), score))
        
        # Automatic differentiation for computing sensitivity
        controls = [Control(g) for g in gamma_funcs]
        rf_cost1 = [ReducedFunctional(cost_comp, c) for c in controls]
        rf_cost2 = [ReducedFunctional(cost_mass, c) for c in controls]

        dfdx_list = [ rf1.derivative().vector().get_local() + rf2.derivative().vector().get_local()
                                                             for rf1, rf2 in zip(rf_cost1, rf_cost2) ]

        ctx.save_for_backward(*[torch.from_numpy(np.array(dfdx)) for dfdx in dfdx_list])

        return gamma_tensor.new_full(gamma_tensor.shape, cost)
              
    @staticmethod
    def backward(ctx, grad_output):
        dfdx_list = ctx.saved_tensors
        grad_gamma_list = [dfdx for dfdx in dfdx_list]
        grad_gamma = torch.stack(grad_gamma_list, dim=0)
        return grad_gamma, None





# define net
net = TopNet(
    numLayers=3,
    numNeuronsPerLyr=50,
    nelx=int(DX*nuel),
    nely=int(DY*nuel),
    symXAxis=False,
    symYAxis=True ) # False

# input coordinate
XY_input = to_torch(Q.tabulate_dof_coordinates().reshape(-1,2)).float()*nuel
XY_input.requires_grad = True

optimizer = torch.optim.Adam(net.parameters(), amsgrad=True, lr=learning_rate)

# set output file 
history_file = open(filename+"history.dat","w")
history_file.write('#%s\t %s\t %s\t %s\t %s\t %s\n' % ("Iter","Cost","compliance","volfrac","grayness","Score"))  # first line

timelog_path = os.path.join(filename, "time_log.txt")
timing_stats = {
    "Forward_net": 0.0,
    "FEA_loss": 0.0,
    "Backward": 0.0,
    "Optim_step": 0.0, }

for epoch in range(1,maxEpochs+1,1):

    optimizer.zero_grad()

    with Timer("Forward_net") as t:
        gamma_b = net(XY_input)
    timing_stats["Forward_net"] += t.interval

    
    with Timer("FEA_loss") as t:
        cost = FEA.apply(gamma_b, epoch)
    timing_stats["FEA_loss"] += t.interval

    
    
    with Timer("Backward") as t:
        cost.mean().backward()
    timing_stats["Backward"] += t.interval

    
    
    with Timer("Optim_step") as t:
        torch.nn.utils.clip_grad_norm_(net.parameters(), nrmThreshold)
        optimizer.step()
    timing_stats["Optim_step"] += t.interval
    
    
    with open("materials.json", "r") as f:
        mat_config = json.load(f)
    E_list = np.array([mat["E"] for mat in mat_config["materials"]])
    num_materials = len(E_list)
    gamma_np = gamma_b.detach().cpu().numpy().T
    E_exp = np.dot(gamma_np, E_list)
    hard_assign_idx = np.array([ np.argmin(np.abs(E_list - e)) for e in E_exp])

    Mf = Function(Q)
    Mf.vector()[:] = hard_assign_idx.astype(float) 
    if epoch % 20 == 0 or epoch == maxEpochs:
        File(filename+str(epoch)+"_gamma_harf_assign.pvd") << Mf
    
    vals = Mf.vector().get_local()
    print("Unique values in f:", np.unique(vals))

    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.colorbar(plot(Mf, cmap="viridis", vmin=0, vmax=num_materials))
    plt.xlim([0, DX])
    plt.ylim([0, DY])
    plt.title("Density Field")
    plt.pause(1)

    # Update multipliers and penalty
    lag_lambda = max(0, lag_lambda+2.0*mass_diff)
    lag_mu = min(massfrac*100, lag_mu*1.2 if abs(mass_diff) > 0.01 else lag_mu)
    lag_mu2 = min(massfrac*100, lag_mu2 + 0.01);
    PenaltyP  = max(3.0, min(PenaltyP*1.1, 5))
    # print(lag_lambda, lag_mu)

with open(timelog_path, "w") as timelog:
    timelog.write("=== Average Time per Iteration ===\n")
    print("\n=== Average Time per Iteration ===")
    for key, total_time in timing_stats.items():
        avg_time = total_time / (maxEpochs + 1)
        line = f"{key:>15}: {avg_time:.6f} s\n"
        print(line.strip())
        timelog.write(line)

exit()

