
from dolfin import *
import numpy as np, sklearn.metrics.pairwise as sp
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix
import time
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0,'../../')

set_log_level(50)

filename = "output_OC/"
os.makedirs(filename, exist_ok=True)

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

# A 55 LINE TOPOLOGY OPTIMIZATION CODE ---------------------------------
def main(DX, DY, volfrac, penal, rmin):
    sigma = lambda _u: 2.0 * mu * sym(grad(_u)) + lmbda * tr(sym(grad(_u))) * Identity(len(_u))
    psi = lambda _u: lmbda / 2 * (tr(sym(grad(_u))) ** 2) + mu * tr(sym(grad(_u)) * sym(grad(_u)))
    mu, lmbda = Constant(0.3846), Constant(0.5769)
    # PREPARE FINITE ELEMENT ANALYSIS ----------------------------------
    mesh = RectangleMesh(Point(0, 0), Point(DX, DY), 150, 50, "right/left")
    U = VectorFunctionSpace(mesh, "P", 1)
    D = FunctionSpace(mesh, "DG", 0)
    D1 = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(U), TestFunction(U)
    u_sol, density, density_old, density_new = Function(U), Function(D, name="density"), Function(D), Function(D)
    density.vector()[:] = 0.5
    V0 = assemble(1 * TestFunction(D) * dx)
    # DEFINE SUPPORT ---------------------------------------------------
    support = CompiledSubDomain("near(x[0], 0.0, tol) && on_boundary", tol=1e-14)
    bcs = [DirichletBC(U, Constant((0.0, 0.0)), support)]
    # DEFINE LOAD ------------------------------------------------------
    load_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    CompiledSubDomain("x[0]==l && x[1]<=0.05", l=DX).mark(load_marker, 1)
    ds = Measure("ds")(subdomain_data=load_marker)
    F = dot(v, Constant((0.0, -0.5))) * ds(1)
    # SET UP THE VARIATIONAL PROBLEM AND SOLVER ------------------------
    K = inner(density ** penal * sigma(u), grad(v)) * dx
    solver = LinearVariationalSolver(LinearVariationalProblem(K, F, u_sol, bcs))
    # PREPARE DISTANCE MATRICES FOR FILTER -----------------------------
    midpoint = [cell.midpoint().array()[:] for cell in cells(mesh)]
    distance_mat = np.maximum(rmin - sp.euclidean_distances(midpoint, midpoint), 0)
    distance_sum = distance_mat.sum(1)
    # START ITERATION --------------------------------------------------
    history_file = open(filename+"history.dat","w")
    history_file.write('#%s\t %s\t %s\t %s\n' % ("Iter","compliance","volfrac","grayness"))
    timelog_path = os.path.join(filename, "time_log.txt")
    
    loop, change, MaxIter = 0, 1, 150
    while loop < MaxIter:
        loop = loop + 1
        with Timer("Forward_net") as t:
            density_old.assign(density)
        timing_stats["Forward_net"] += t.interval
        
        with Timer("FEA_loss") as t:
            solver.solve()
            objective = density ** penal * psi(u_sol)
        timing_stats["FEA_loss"] += t.interval

        with Timer("Backward") as t:
            sensitivity = project(-diff(objective, density), D).vector()[:]
        timing_stats["Backward"] += t.interval
        
        with Timer("Forward_net") as t:
            sensitivity = np.divide(distance_mat @ np.multiply(density.vector()[:], sensitivity), np.multiply(density.vector()[:], distance_sum))
        timing_stats["Forward_net"] += t.interval
        
        
        with Timer("Optim_step") as t:
            l1, l2, move = 0, 100000, 0.01
            while l2 - l1 > 1e-4:
                l_mid = 0.5 * (l2 + l1)
                density_new.vector()[:] = np.maximum(0.001, np.maximum(density.vector()[:] - move, np.minimum(1.0, np.minimum(density.vector()[:] + move, density.vector()[:] * np.sqrt(-sensitivity / V0 / l_mid)))))
                current_vol = assemble(density_new * dx)
                l1, l2 = (l_mid, l2) if current_vol > volfrac * V0.sum() else (l1, l_mid)
        timing_stats["Optim_step"] += t.interval
        
        volume = DX*DY
        grayness = assemble(4*density_new*(1-density_new)*dx)/volume
        change = abs(max(density_new.vector()[:] - density_old.vector()[:]))
        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(loop, project(objective, D).vector().sum(), current_vol / V0.sum(), change))
        density.assign(density_new)
        history_file.write('%s\t %s\t %s\t %s\n' % (loop, project(objective, D).vector().sum(), current_vol/V0.sum(), grayness))
    
        if loop%20==0 or loop == MaxIter:
            File(filename+str(loop)+"_gamma.pvd") << project(density,D1)
        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.colorbar(plot(density, cmap="viridis", vmin=density.vector().get_local().min(), vmax=density.vector().get_local().max()))
        plt.xlim([0, DX])
        plt.ylim([0, DY])
        plt.title("Density Field")
        plt.pause(1)

    with open(timelog_path, "w") as timelog:
        timelog.write("=== Average Time per Iteration ===\n")
        print("\n=== Average Time per Iteration ===")
        for key, total_time in timing_stats.items():
            avg_time = total_time / (MaxIter + 1)
            line = f"{key:>15}: {avg_time:.6f} s\n"
            print(line.strip())
            timelog.write(line)


# The real main driver
if __name__ == "__main__":
    main(DX=3, DY=1, volfrac=0.4, penal=5.0, rmin=0.07)












