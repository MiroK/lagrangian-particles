from LagrangianParticles import LagrangianParticles
from particle_generators import RandomCircle
import matplotlib.pyplot as plt
from dolfin import VectorFunctionSpace, interpolate, RectangleMesh, Expression
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD

mesh = RectangleMesh(0, 0, 1, 1, 10, 10)
particle_positions = RandomCircle([0.5, 0.75], 0.15).generate([100, 100])

V = VectorFunctionSpace(mesh, 'CG', 1)
u = interpolate(Expression(("-2*sin(pi*x[1])*cos(pi*x[1])*pow(sin(pi*x[0]),2)",
                            "2*sin(pi*x[0])*cos(pi*x[0])*pow(sin(pi*x[1]),2)")),
                V)
lp = LagrangianParticles(V)
lp.add_particles(particle_positions)

fig = plt.figure()
lp.scatter(fig)
fig.suptitle('Initial')

if comm.Get_rank() == 0:
    fig.show()

plt.ion()

dt = 0.01
for step in range(500):
    lp.step(u, dt=dt)

    lp.scatter(fig)
    fig.suptitle('At step %d' % step)
    fig.canvas.draw()
    fig.clf()