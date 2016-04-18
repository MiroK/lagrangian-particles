from LagrangianParticles import LagrangianParticles
from particle_generators import RandomRectangle
from dolfin import VectorFunctionSpace, interpolate, Expression, RectangleMesh, Point
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI as pyMPI
import matplotlib.colors as colors
import matplotlib.cm as cmx

comm = pyMPI.COMM_WORLD

mesh = RectangleMesh(Point(0, 0), Point(1, 1), 10, 10)
particle_positions = RandomRectangle(Point(0.125, 0.25), Point(0.75, 0.8)).generate([100,
                                                                       100])

V = VectorFunctionSpace(mesh, 'CG', 1)
u = interpolate(Expression(("-2*sin(pi*x[1])*cos(pi*x[1])*pow(sin(pi*x[0]),2)",
                            "2*sin(pi*x[0])*cos(pi*x[0])*pow(sin(pi*x[1]),2)")),
                V)
lp = LagrangianParticles(V)
lp.add_particles(particle_positions)

# Initialize scatter plot
fig0 = plt.figure()
lp.scatter(fig0)
fig0.suptitle('Initial')
# Initialize bar plot
fig1 = plt.figure()
lp.bar(fig1)
fig1.suptitle('Initial')

plt.ion()

# Data for comunicating cpu time and particles count
my_cpu = np.zeros(1, 'float')
all_cpu = np.zeros(comm.Get_size(), 'float')

my_count = np.zeros(1, 'I')
all_count = np.zeros(comm.Get_size(), 'I')

# Steps to be used in computation
steps = np.arange(500)

if comm.Get_rank() == 0:
    procs = range(comm.Get_size())
    # Data for history of cpu time per step and particle count
    cpu_histories = np.zeros((len(steps), len(procs)))
    count_histories = np.zeros_like(cpu_histories)
    # Similar to scatter we color history curve by proces it belongs to
    cmap = cmx.get_cmap('jet')
    cnorm = colors.Normalize(vmin=0, vmax=len(procs))
    scalarMap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
    # Two subplots, one for histories of cpu time, the other for particle
    # count
    fig2, axarr = plt.subplots(2, sharex=True)
    lines = [[axarr[i].plot(steps,
                            steps,
                            label='%d' % proc,
                            c=scalarMap.to_rgba(proc))[0]
              for proc in procs]
             for i in range(2)]
    # Guess this should be enough for time step
    axarr[0].set_ylim([-0.1, 2])
    axarr[0].set_ylabel('s/step')
    # We share axes so remove labels of x-axies
    plt.setp([a.get_xticklabels() for a in fig2.axes[:0]], visible=False)
    # Particle loaf handles the x-axes
    axarr[1].set_ylim([-10, len(particle_positions)+10])
    axarr[1].set_ylabel('number of particles')
    axarr[1].set_xlabel('step')
    # One legend shared for both plots
    axarr[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                    ncol=len(procs))
    # Put plots togeter
    fig2.subplots_adjust(hspace=0.05)

    fig0.show()
    fig1.show()
    fig2.show()

dt = 0.01
for step in steps:
    # Communicate the cpu time
    my_cpu[0] = sum(lp.step(u, dt=dt))
    comm.Gather(my_cpu, all_cpu, root=0)
    # Communicate particle load
    my_count[0] = lp.total_number_of_particles()[1]
    comm.Gather(my_count, all_count, root=0)

    # Update history plot
    if comm.Get_rank() == 0:

        cpu_histories[step, :] = all_cpu
        count_histories[step, :] = all_count

        for proc in procs:
            lines[0][proc].set_ydata(cpu_histories[:, proc])

        for proc in procs:
            lines[1][proc].set_ydata(count_histories[:, proc])

        fig2.canvas.draw()

    # Update scatter
    lp.scatter(fig0)
    fig0.suptitle('At step %d' % step)

    # Update bar plot
    n_particles = lp.bar(fig1)
    if n_particles is not None:
        fig1.suptitle('At step %d, total particles %d' % (step, n_particles))

    fig0.canvas.draw()
    fig0.clf()
    fig1.canvas.draw()
    fig1.clf()

# Save
if comm.Get_rank() == 0:
    fig2.savefig('history.pdf')
