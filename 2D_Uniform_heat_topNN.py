
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
import os


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



filename = "output_Uniform_heat/"
os.makedirs(filename, exist_ok=True)

# -----------------------------------------
DX, DY, nuel = 2.0, 1.0, 70
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
lag_lambda, lag_mu, lag_mu2 = 0., 10., 10.  # Lagrange penalty parameter
maxEpochs, learning_rate, nrmThreshold = 118, 0.01, 10  # nrmThreshold for gradient clipping

# Topopt user setting
ft = 0     # 0:net       1:net+pde        2:net+hv          3:net+pde+hv
volfrac = 0.3
useIterative = 0

# PDE setting
filterR = 7/nuel
print('filter type:', ft)
print("filter radius:", filterR)

# Heaviside setting
eta = 0.5
density_beta_beginning = 100
density_beta_step = 40
density_beta = 1.0
density_betaIncScale = 2.0
density_betamax = 16

PenaltyP_E, PenaltyR_E, RAMP_E = 3.0, 5.0, 0
PenaltyP_K, PenaltyR_K, RAMP_K = 3.0, 5.0, 0

T0 = 30

E0, Emin = 1, 1e-6
K0, Kmin = 1, 1e-3
alpha = 7e-6

# Definition of the Lame's parameters
nu = 0.3
lmbda, mu = 1.0*nu/((1 + nu)*(1 - 2*nu)), 1.0/(2*(1 + nu))

# domain markers 
optDom, voidDom, solidDom = [], [0], [1]

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

class Sine(nn.Module):
    def forward(self, x):
        #return torch.sin(x)
        return 2*torch.sin(x)*torch.cos(x)

#%% Neural network
class TopNet(nn.Module):
    inputDim = 2; # x and y coordn of the point
    outputDim = 2; # if material/void at the point
    def __init__(self, numLayers, numNeuronsPerLyr, nelx, nely, symXAxis, symYAxis):
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
            # nn.init.kaiming_normal_(l.weight, nonlinearity='leaky_relu');  # "he"
            # nn.init.uniform_(l.weight, a=-0.1, b=0.1);            # "uniform"
            # nn.init.normal_(l.weight, mean=0.0, std=0.05)         # "normal"

            nn.init.zeros_(l.bias);
            self.layers.append(l);
            current_dim = numNeuronsPerLyr;
        self.layers.append(nn.Linear(current_dim, self.outputDim));
        self.bnLayer = nn.ModuleList();
        for lyr in range(numLayers): # batch norm 
            self.bnLayer.append(nn.BatchNorm1d(numNeuronsPerLyr));
    def forward(self, x, fixedIdx = None):
        m = Sine()
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
        out = torch.softmax(self.layers[-1](x), dim = 1); # output layer
        rho_material1 = out[:,0].view(-1); # grab only the first output
        rho_material2 = out[:,1].view(-1); # grab only the first output

        return  rho_material1;
        





class PDEfilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gamma_tensor):

        # input density tensor
        gamma = Function(Q)
        gamma.vector()[:] = gamma_tensor.detach().cpu().numpy().reshape(-1)

        # Create trial and test functions
        v = TrialFunction(Q)
        dv = TestFunction(Q)

        # Construct bilinear form
        a = filterR/(2*sqrt(3))*inner(grad(v), grad(dv))*dx + dot(v, dv)*dx

        # Construct linear form
        L = dot(gamma, dv)*dx

        filter_A, filter_b = assemble_system(a, L, [])
        
        # Solve the equation
        gamma_f = Function(Q)
        solve(filter_A, gamma_f.vector(), filter_b, 'mumps')

        # save grad to ctx
        ctx.filter_A = filter_A
        ctx.filter_b = filter_b
        ctx.gamma_tensor = gamma_tensor

        # Return the solution
        return torch.from_numpy(gamma_f.vector().get_local().reshape(gamma_tensor.shape))

    @staticmethod
    def backward(ctx, grad_output):
        # 
        grad_fun = Function(Q)
        grad_fun.vector()[:] = grad_output.cpu().numpy().reshape(-1)
        
        b_adj = Function(Q)
        b_adj.vector()[:] = ctx.filter_b * grad_fun.vector()

        # 
        grad_filter = Function(Q)
        df.solve(ctx.filter_A, grad_filter.vector(), b_adj.vector(), 'mumps')
        
        grad_gamma = ctx.filter_b * grad_filter.vector()
        
        return torch.from_numpy(grad_gamma.get_local().reshape(ctx.gamma_tensor.shape))




# HV filter
def HVfilter(gamma):
    # gamma: torch.Tensor
    beta = torch.tensor(density_beta, dtype=gamma.dtype, device=gamma.device)
    eta_ = torch.tensor(eta, dtype=gamma.dtype, device=gamma.device)

    numerator = torch.tanh(beta * eta_) + torch.tanh(beta * (gamma - eta_))
    denominator = torch.tanh(beta * eta_) + torch.tanh(beta * (1.0 - eta_))
    return numerator / denominator



# -------------------- subdomains setting ----------------------------------------- 
TOL = 1e-5

GLOBAL_DOF = 0
def SetUpOptRegions(mesh,Q,domains,optDom,voidDom,solidDom):
    
    optIdx = []   # index of optimization region
    voidIdx = []  # index of passive void region
    solidIdx = [] # index of passive solid region
   
    if (voidDom and solidDom) is None: # there are no passive regions
        optIdx = range(0,Q.dim())
    else:
        # Define design domain
        dofmap = Q.dofmap()
        my_first, my_last = dofmap.ownership_range()                # global
        unowned = dofmap.local_to_global_unowned()
        # index of sio2 spheres
        for cell in cells(mesh): # compute dofs in the domains
            cell_dofs = dofmap.cell_dofs(cell.index())  
            local_dofs = []
            global_dofs = []
            for dof in cell_dofs:
                global_dof = dofmap.local_to_global_index(dof)  # global
                if my_first <= global_dof < my_last:
                    local_dofs.append(dof)
                    global_dofs.append(global_dof)
            if domains[cell] in optDom:
                if GLOBAL_DOF:
                    optIdx.extend(global_dofs)
                else:
                    optIdx.extend(local_dofs)
            elif domains[cell] in voidDom:
                if GLOBAL_DOF:
                    voidIdx.extend(global_dofs)
                else:
                    voidIdx.extend(local_dofs)
            else:
                if GLOBAL_DOF:
                    solidIdx.extend(global_dofs)
                else:
                    solidIdx.extend(local_dofs)

        # unique
        voidIdx = list(set(voidIdx))
        solidIdx = list(set(solidIdx))
        optIdx = list(set(optIdx))

    return optIdx,voidIdx,solidIdx


def SIMP_E(gamma, T):
    # E depended on T, T:10-->40 E:E0-->0.3*E0
    ET = (0.1*E0 - E0) * T / (40.0 - 10.0) + ( E0 - (0.1*E0 - E0) * 10 / (40.0 - 10.0) )
    if RAMP_E:
        return Emin + (ET - Emin) * gamma / (1. + PenaltyR_E * (1. - gamma))
    else:
        return Emin+(ET-Emin)*gamma**PenaltyP_E

def SIMP_K(gamma):
    if RAMP_K:
        return Kmin + (K0 - Kmin) * gamma / (1. + PenaltyR_K * (1. - gamma))
    else:
        return Kmin+(K0-Kmin)*gamma**PenaltyP_K

# Mechanical stress
def eps(v):
    return sym(grad(v))
def sigma(v, dT):
    return (lmbda*tr(eps(v))- alpha*(3*lmbda+2*mu)*(dT-T0))*Identity(2) + 2.0*mu*eps(v)

def sigma_mech(v): 
    return lmbda*tr(eps(v))*Identity(2) + 2.0*mu*eps(v)
    
class Heat_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0) and on_boundary

class Force_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return 0 <= x[1] <= 0.1 and ( DX/2-0.05 <= x[0] <= DX/2+0.05 or DX/4-0.05 <= x[0] <= DX/4+0.05 or 3*DX/4-0.05 <= x[0] <= 3*DX/4+0.05)

class Fixed_Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return  ( near(x[0], 0) or near(x[0], DX) )and on_boundary

# Solve the FE problem
class FEA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gamma_tensor, loop_iter):

        # input density tensor
        gamma = Function(Q)
        gamma.vector()[:] = gamma_tensor.detach().cpu().numpy().reshape(-1)
        
        # Define passive region
        passive = Force_boundary()
        domains = MeshFunction("size_t",mesh,mesh.topology().dim())
        domains.set_all(0)
        passive.mark(domains, 1)
        global passiveIdx
        optIdx, voidIdx, passiveIdx = SetUpOptRegions(mesh,Q,domains,optDom,voidDom,solidDom)
        #gamma.vector()[passiveIdx] = 1

        # load region
        force_Load_domain = Force_boundary()
        heat_Load_domain = Heat_boundary()
        domains = MeshFunction("size_t",mesh,mesh.topology().dim())
        domains.set_all(0)
        force_Load_domain.mark(domains, 1)
        heat_Load_domain.mark(domains, 2)
        dx = Measure("dx")(subdomain_data=domains)
        ds = Measure("ds")(subdomain_data=domains)

        # Boundary condition
        th_bcs = [DirichletBC(Q, Constant(40), Heat_boundary()),
        DirichletBC(Q, Constant(10), Fixed_Boundary())]

        # 1) Thermal problem
        T_ = TrialFunction(Q)
        tT = TestFunction(Q)
        Qf = Constant(0)
        #Qf = Expression("sin(pi*x[0])*cos(pi*x[1])", degree=3)

        a = SIMP_K(gamma)*df.inner(grad(T_), grad(tT))*dx
        L = df.inner(tT,Qf)*ds(2)
        T = Function(Q)

        heat_A, heat_b = assemble_system(a, L, th_bcs)
        with Timer() as t:
            if useIterative==0:
                solve(heat_A, T.vector(), heat_b, 'mumps')
            else:
                solver3_setting(1e-12)
                solver3.solve(heat_A, T.vector(), heat_b)

        if loop_iter == maxEpochs or loop_iter%1 == 0:
            File(filename+str(loop_iter)+"Final_T.pvd") << T
            File(filename+str(loop_iter)+"Final_E.pvd") << project( SIMP_E(gamma, T), Q)
            

        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.colorbar(plot(T, cmap="viridis", vmin=T.vector().get_local().min(), vmax=T.vector().get_local().max()))
        plt.xlim([0, DX])
        plt.ylim([0, DY])
        plt.title("T")
        plt.pause(1)

        # 2) linear elasticity problem
        El_bcs = [DirichletBC(V, Constant((0,0)), Fixed_Boundary())]
        
        # Create trial and test functions
        u_ = TrialFunction(V)
        tu = TestFunction(V)
        f = Constant((0, -5.0))

        F = SIMP_E(gamma, T)*df.inner(sigma(u_, T), eps(tu))*dx - df.inner(f,tu)*ds(1)
        Dis = Function(V)
        elasticity_A, elasticity_b = lhs(F), rhs(F)
    
        solve(elasticity_A == elasticity_b, Dis, El_bcs)

        # obj function
        compliance = assemble(SIMP_E(gamma, T)*inner(sigma_mech(Dis), eps(Dis))*dx)
        vol        = assemble(gamma*dx)
        global vol_diff
        vol_diff   = vol/volume-volfrac

        grayness = assemble(4*gamma*(1-gamma)*dx)/volume
       
        if loop_iter == 1:
            global compliance0
            compliance0 = assemble(SIMP_E(gamma, T)*inner(sigma_mech(Dis), eps(Dis))*dx)
            #assemble(SIMP_E(interpolate(Constant(volfrac),Q),T)*df.inner(sigma(Dis, T), eps(Dis))*dx) 

        score = 0.5*compliance/compliance0 + 0.3*grayness/0.01 + 0.2*loop_iter/maxEpochs
        

        cost = compliance/compliance0 + lag_lambda*vol_diff + 0.5*lag_mu*vol_diff**2
        #cost = compliance/compliance0 + lag_mu2*vol_diff**2
        print("Iter:", loop_iter, "cost:", cost, "compliance:", compliance, "vol:", vol/volume, "gray:", grayness, "score:", score)

        history_file.write('%s\t %s\t %s\t %s\t %s\t %s\n' % (loop_iter, cost, compliance, vol/volume, grayness, score))
        
        # Automatic differentiation for computing sensitivity
        control = Control(gamma)
        rf_cost = ReducedFunctional(cost, control)
        dfdx = rf_cost.derivative().vector().get_local()
        
        # save grad to ctx
        ctx.save_for_backward(torch.from_numpy(dfdx).to(gamma_tensor.device))
        
        return gamma_tensor.new_full(gamma_tensor.shape, cost)
              
    @staticmethod
    def backward(ctx, grad_output):
        (dcost_dgamma,) = ctx.saved_tensors
        return grad_output * dcost_dgamma, None




# define net
net = TopNet(
    numLayers=5,
    numNeuronsPerLyr=40,
    nelx=int(DX*nuel),
    nely=int(DY*nuel),
    symXAxis=True,  # True
    symYAxis=False ) # False

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
        if ft == 0:
            gamma_b = net(XY_input)
        elif ft == 1:
            gamma = net(XY_input)    
            gamma_b = PDEfilter.apply(gamma)
        elif ft == 2:
            gamma = net(XY_input)
            gamma_b = HVfilter(gamma)
        elif ft == 3:
            gamma = net(XY_input)    
            gamma_f = PDEfilter.apply(gamma)
            gamma_b = HVfilter(gamma_f)
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
    
    
    
    gamma_out = Function(Q)
    gamma_out.vector()[:] = to_np(gamma_b).reshape(-1)
    if epoch % 20 == 0 or epoch == maxEpochs:
        File(filename+str(epoch)+"_gamma.pvd") << gamma_out

    plt.ion()
    plt.figure(2)
    plt.clf()
    plt.colorbar(plot(gamma_out, cmap="viridis", vmin=gamma_out.vector().get_local().min(), vmax=gamma_out.vector().get_local().max()))
    plt.xlim([0, DX])
    plt.ylim([0, DY])
    plt.title("Density Field")
    plt.pause(1)

    # Update multipliers and penalty
    lag_lambda = max(0, lag_lambda+vol_diff)
    lag_mu = min(volfrac*100, (lag_mu+0.01) if abs(vol_diff) > 0.01 else lag_mu)
    lag_mu2 = min(volfrac*100, lag_mu2 + 0.01);
    PenaltyP_E, PenaltyR_E  = min(3.0, PenaltyP_E + 0.01), min(5.0, PenaltyR_E + 0.01)
    PenaltyP_K, PenaltyR_K  = min(3.0, PenaltyP_K + 0.01), min(5.0, PenaltyR_K + 0.01)

    # Heaviside beta update
    if epoch > density_beta_beginning and epoch % density_beta_step == 0 and ft >= 2 and density_beta < density_betamax:
        density_beta = min(density_beta * density_betaIncScale, density_betamax)

with open(timelog_path, "w") as timelog:
    timelog.write("=== Average Time per Iteration ===\n")
    print("\n=== Average Time per Iteration ===")
    for key, total_time in timing_stats.items():
        avg_time = total_time / (maxEpochs + 1)
        line = f"{key:>15}: {avg_time:.6f} s\n"
        print(line.strip())
        timelog.write(line)

exit()

