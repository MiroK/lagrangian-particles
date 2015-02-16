from LagrangianParticles import LagrangianParticles, ParticleSource
from particle_generators import RandomCircle
import matplotlib.pyplot as plt
from dolfin import VectorFunctionSpace, interpolate, RectangleMesh, Expression,\
    plot, interactive, Function, FunctionSpace, CompiledSubDomain, Constant
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD

mesh = RectangleMesh(0, 0, 1, 1, 50, 50)
particle_positions = RandomCircle([0.5, 0.75], 0.15).generate([100, 100])
#print particle_positions
#print type(particle_positions)

V = VectorFunctionSpace(mesh, 'CG', 1)
u = interpolate(Expression(("-2*sin(pi*x[1])*cos(pi*x[1])*pow(sin(pi*x[0]),2)",
                            "2*sin(pi*x[0])*cos(pi*x[0])*pow(sin(pi*x[1]),2)")),
                V)
#u = Constant(1.0)
S = FunctionSpace(mesh, 'DG', 0)
rho = Function(S)
c = Function(S)

lp = LagrangianParticles(V)
#lp.add_particles(particle_positions)

circ = CompiledSubDomain("(x[0]-0.5)*(x[0]-0.5) + (x[1]-0.75)*(x[1]-0.75) < 0.1*0.1")

source = ParticleSource(100, circ, mesh, lp)
source.apply_source()


fig = plt.figure()
lp.scatter(fig)
fig.suptitle('Initial')
fig.clf()

if comm.Get_rank() == 0:
    fig.show()

plt.ion()


lp.particle_density(rho)
#plot(rho, title='0')

dt = 0.01
for step in range(100):
    lp.step(u, dt=dt)
    if step % 10 == 0:
    	lp.scatter(fig)
        fig.suptitle('At step %d' % step)
        fig.canvas.draw()
        fig.clf()
    source.apply_source()

    lp.particle_density(rho)
    #plot(rho, title='%d' % step)

interactive()
