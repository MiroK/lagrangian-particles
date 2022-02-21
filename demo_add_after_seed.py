from LagrangianParticles import LagrangianParticles
from particle_generators import RandomCircle
import matplotlib.pyplot as plt
from dolfin import VectorFunctionSpace, interpolate, RectangleMesh, Expression, Point
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD

mesh = RectangleMesh(Point(0, 0), Point(1, 1), 10, 10)
particle_positions = RandomCircle([0.5, 0.75], 0.15).generate([100, 100])
# NOTE: Here we just have record the original particles and in the stepping
# we will add them for the purpose of the demo
old_ones = 1*particle_positions

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

save = False

dt = 0.01
for step in range(500):
    lp.step(u, dt=dt)
    # Add the old ones. Note that 1*old_ones is a new array; we do this
    # to have new particles as pointer to new data rather than new pointers
    # to old data
    lp.add_particles(1*old_ones)

    print(lp.total_number_of_particles(), '<<< particles')
    
    lp.scatter(fig)
    fig.suptitle('At step %d' % step)
    fig.canvas.draw()

    if save: plt.savefig('img%s.png' % str(step).zfill(4))

    fig.clf()
