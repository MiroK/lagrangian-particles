from LagrangianParticles import LagrangianParticles
from particle_generators import RandomCircle, circle
import matplotlib.pyplot as plt
from dolfin import VectorFunctionSpace, interpolate, RectangleMesh, Expression, Function
from mpi4py import MPI as pyMPI
import time

comm = pyMPI.COMM_WORLD

mesh = RectangleMesh(0, 0, 1, 1, 20, 20)
#particle_positions = RandomCircle([0.5, 0.75], 0.15).generate([50, 50])
particle_positions = circle([0.5, 0.75], 0.15, 200)


V = VectorFunctionSpace(mesh, 'CG', 1)
u_plus = interpolate(Expression(("-2*sin(pi*x[1])*cos(pi*x[1])*pow(sin(pi*x[0]),2)",
                            "2*sin(pi*x[0])*cos(pi*x[0])*pow(sin(pi*x[1]),2)")),
                V)

u_minus = interpolate(Expression(("2*sin(pi*x[1])*cos(pi*x[1])*pow(sin(pi*x[0]),2)",
                            "-2*sin(pi*x[0])*cos(pi*x[0])*pow(sin(pi*x[1]),2)")),
                V)
lp = LagrangianParticles(V)
lp.add_particles(particle_positions)

fig0 = plt.figure()
lp.scatter(fig0)
fig0.suptitle('Initial')

fig = plt.figure()

if comm.Get_rank() == 0:
    fig0.show()


#plt.ion()

dt = 0.001
T = 20.0
step = 0
t = 0
if comm.Get_rank() == 0:
    t0 = time.time()
while t <= T:
    step += 1
    t = step*dt
    if comm.Get_rank() == 0:
        print "t =", t
    if t<=10.0:
        u = u_plus
    else:
        u = u_minus
    
    lp.step(u,dt)

    if step % 100 == 0:
        lp.scatter(fig)
        fig.suptitle('At step %d' % step)
        fig.canvas.draw()
        fig.clf()

if comm.Get_rank() == 0:
    t1 = time.time()
    print "Total time: ", t1-t0

lp.scatter(fig)
if comm.Get_rank() == 0:
    fig.show()
raw_input("Press enter to exit...")
