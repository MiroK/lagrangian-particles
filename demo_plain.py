from LagrangianParticles import LagrangianParticles
from particle_generators import RandomCircle
import matplotlib.pyplot as plt
from boundary_collisions import remove_particles
from dolfin import VectorFunctionSpace, interpolate, RectangleMesh, Expression, Point
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD

mesh = RectangleMesh(Point(0, 0), Point(1, 1), 10, 10)
particle_positions = RandomCircle([0.5, 0.75], 0.15).generate([100, 100])

V = VectorFunctionSpace(mesh, 'CG', 1)
u = interpolate(Expression(("-2*sin(pi*x[1])*cos(pi*x[1])*pow(sin(pi*x[0]),2)",
                            "2*sin(pi*x[0])*cos(pi*x[0])*pow(sin(pi*x[1]),2)"), degree=4),
                V)
lp = LagrangianParticles(V)
lp.add_particles(particle_positions)

fig = plt.figure()
lp.scatter(fig)
fig.suptitle('Initial')

if comm.Get_rank() == 0:
    fig.show()

plt.ion()

# Remove particles based on ...
predicates = (lambda x: x[1] < 0.5 and x[0] > 0.75, )

save = False

dt = 0.01
for step in range(500):
    lp.step(u, dt=dt)

    lp = remove_particles(lp, predicates)

    lp.scatter(fig)
    fig.suptitle('At step %d' % step)
    fig.canvas.draw()

    if save: plt.savefig('img%s.png' % str(step).zfill(4))

    fig.clf()
