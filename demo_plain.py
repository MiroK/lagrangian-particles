from LagrangianParticles import LagrangianParticles
from particle_generators import RandomCircle
import matplotlib.pyplot as plt
from dolfin import *
from mpi4py import MPI as pyMPI
import numpy as np

comm = pyMPI.COMM_WORLD

mesh = RectangleMesh(0, 0, 1, 1,50,50)
#mesh = Mesh("squre.xml")

def rectangle(x0,x1,y0,y1,Nx,Ny):
    lst = list()
    xs = np.linspace(x0,x1,Nx)
    ys = np.linspace(y0,y1,Ny)
    for y in ys:
        lst += [np.array([x, y]) for x in xs]
    
    return lst
  

#particle_positions = rectangle(0.45,0.55,0,0.1,100,100)

particle_positions = RandomCircle([0.5, 0.75], 0.15).generate([200, 200])

V = VectorFunctionSpace(mesh, 'CG', 1)
u = interpolate(Expression(("-2*sin(pi*x[1])*cos(pi*x[1])*pow(sin(pi*x[0]),2)",
                            "2*sin(pi*x[0])*cos(pi*x[0])*pow(sin(pi*x[1]),2)")),
                V)

#u = interpolate(Expression(("(x[0] - 0.5)", "0.0")), V)
#u = Constant((1.0, 0.0))
lp = LagrangianParticles(V)
lp.add_particles(particle_positions)

fig = plt.figure()
lp.scatter(fig)
fig.suptitle('Initial')

if comm.Get_rank() == 0:
    fig.show()

plt.ion()

DG0 = FunctionSpace(mesh, "DG", 0)
CG1 = FunctionSpace(mesh, "CG", 1)
rho = Function(DG0)
rhofile = File("results/rhofile.pvd")
dt = 0.01
for step in range(500):
    #print "Step: ",step
    #import time
    #time.sleep(1)
    lp.step(u, dt=dt)
    lp.scatter(fig)
    fig.suptitle('At step %d' % step)
    fig.canvas.draw()
    fig.clf()
    lp.update_density(step)
    #if step == 0:
    #    maxrho = max(lp.rho.vector().array())
    #rho.assign(interpolate(rho,CG1))
    #lp.rho.vector()[:] = rho.vector().array()/lp.maxrho
    #rhofile << lp.rho
    plot(lp.rho, rescale=True)
