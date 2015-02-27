from LagrangianParticles import LagrangianParticles, ParticleSource
from particle_generators import RandomCircle, RandomSphere, circle
import matplotlib.pyplot as plt
from dolfin import VectorFunctionSpace, interpolate, RectangleMesh, Expression,\
    plot, interactive, Function, FunctionSpace, CompiledSubDomain, Constant, UnitCubeMesh, Mesh, BoxMesh
from mpi4py import MPI as pyMPI
from numpy import sqrt
from math import erf

comm = pyMPI.COMM_WORLD

#mesh = RectangleMesh(0, 0, 1, 1, 50, 50)
mesh = Mesh("UnitSquare.xml")
#mesh = BoxMesh(0,0,0,0.1,0.1,0.1,50,50,50)
#particle_positions = RandomSphere([0.5, 0.75, 0.5], 0.15).generate([20,20,20], method="full")
#print particle_positions
#print type(particle_positions)

V = VectorFunctionSpace(mesh, 'CG', 1)
#u = interpolate(Expression(("-2*sin(pi*x[1])*cos(pi*x[1])*pow(sin(pi*x[0]),2)",
#                            "2*sin(pi*x[0])*cos(pi*x[0])*pow(sin(pi*x[1]),2)")),
#                V)
#u_exp = Expression(("sin(2*pi*t)", "0.0", "0.0"), t=0.0)
u_exp = Expression(("sin(2*pi*t)", "0.0"), t=0.0)
u = interpolate(u_exp, V)
S = FunctionSpace(mesh, 'DG', 0)
rho = Function(S)
c = Function(S)

lp = LagrangianParticles(V)
#lp.add_particles(particle_positions)

#def concentration_factor(x):
#    r = sqrt((x[0] - 0.009)**2 + (x[1] - 0.06)**2 + (x[2] - 0.003)**2)
#    return 0.1*(1.0 - erf((r - 0.0015)/0.005))
def concentration_factor(x):
    r = sqrt((x[0] - 0.5)**2 + (x[1] - 0.75)**2)
    return 0.1*(1.0 - erf((r - 0.1)/0.03))

circ = CompiledSubDomain("(x[0]-0.5)*(x[0]-0.5) + (x[1]-0.75)*(x[1]-0.75) < 0.5*0.5")
sphere = CompiledSubDomain("(x[0]-0.009)*(x[0]-0.009) + (x[1]-0.06)*(x[1]-0.06) + (x[2]-0.003)*(x[2]-0.003) < 0.01*0.01")
source = ParticleSource(30, circ, mesh, lp, concentration_factor)
source.apply_source()


fig = plt.figure()
lp.scatter(fig, dim=2)
fig.suptitle('Initial')
fig.clf()

if comm.Get_rank() == 0:
    fig.show()

plt.ion()


#lp.particle_density(rho, 0)
#plot(rho, title='0')

dt = 0.01
for step in range(100):
    u_exp.t = step*dt
    u.assign(interpolate(u_exp, V))
    lp.step(u, dt=dt)
    lp.scatter(fig, dim=2)
    fig.suptitle('At step %d' % step)     
    fig.canvas.draw()
    fig.clf()
    source.apply_source()

    lp.particle_density(rho, step)
    plot(rho, title='%d' % step)

plt.ioff()
interactive()
