import numpy as np
from math import pi, sqrt, pow
from itertools import product
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD

def circle(center, radius, N, fill=0):
    '''
    With fill ==0:
        put N particles on circle with radius centered at center
    othersise:
        fill circle with smaller concentric circles
    '''
    if fill == 0:
        theta = np.linspace(0, 2.*pi, N, endpoint=False)
        xs = center[0] + radius*np.cos(theta)
        ys = center[1] + radius*np.sin(theta)
        return [np.array([x, y]) for x, y, in zip(xs, ys)]
    else:
        return sum((circle(center, r, N, fill=0)
                    for r in np.linspace(0.01, radius, fill)), [])


class RandomGenerator(object):
    '''
    Fill object by random points.
    '''
    def __init__(self, domain, rule):
        '''
        Domain specifies bounding box for the shape and is used to generate
        points. The rule filter points of inside the bounding box that are
        axctually inside the shape.
        '''
        assert isinstance(domain, list)
        self.domain = domain
        self.rule = rule
        self.dim = len(domain)
        self.rank = comm.Get_rank()

    def generate(self, N, method='full', seed=False):
        'Genererate points.'
        assert len(N) == self.dim
        assert method in ['full', 'tensor', 'uniform']
        if seed != False:
            np.random.seed(seed)
        if self.rank == 0:
            # Generate random points for all coordinates
            if method == 'full':
                n_points = np.product(N)
                points = np.random.rand(n_points, self.dim)
                for i, (a, b) in enumerate(self.domain):
                    points[:, i] = a + points[:, i]*(b-a)
            # Create points by tensor product of intervals
            elif method == 'tensor':
                # Values from [0, 1) used to create points between
                # a, b - boundary
                # points in each of the directiosn
                shifts_i = np.array([np.random.rand(n) for n in N])
                # Create candidates for each directions
                points_i = (a+shifts_i[i]*(b-a)
                            for i, (a, b) in enumerate(self.domain))
                # Cartesian product of directions yield n-d points
                points = (np.array(point) for point in product(*points_i))
            else:
                points = []
                xs = np.linspace(self.domain[0][0], self.domain[0][1], N[0])
                ys = np.linspace(self.domain[1][0], self.domain[1][1], N[1])
                if self.dim == 3:
                    zs = np.linspace(self.domain[2][0], self.domain[2][1], N[2])
                
                points = []
                if self.dim == 2:
                    for y in ys:
                        points += [[x, y] for x in xs]
                elif self.dim == 3:
                    for z in zs:
                        for y in ys:
                            points += [[x, y, z] for x in xs]


            # Use rule to see which points are inside
            points_inside = np.array(filter(self.rule, points))
        else:
            points_inside = None

        points_inside = comm.bcast(points_inside, root=0)

        return points_inside


class RandomRectangle(RandomGenerator):
    def __init__(self, ax, bx, ay, by):
        assert ax < bx and ay < by
        RandomGenerator.__init__(self, [[ax, bx], [ay, by]], lambda x: True)


class RandomCircle(RandomGenerator):
    def __init__(self, center, radius):
        assert radius > 0
        domain = [[center[0]-radius, center[0]+radius],
                  [center[1]-radius, center[1]+radius]]
        RandomGenerator.__init__(self, domain,
                                 lambda x: sqrt((x[0]-center[0])**2 +
                                                (x[1]-center[1])**2) < radius
                                 )
class RandomCylinder(RandomGenerator):
    def __init__(self, center, radius, height):
        assert radius > 0
        domain = [[center[0]-radius, center[0]+radius],
                 [center[1] -height/2., center[1] + height/2.],
                 [center[2]-radius, center[2]+radius]]
        RandomGenerator.__init__(self, domain,
                                 lambda x: sqrt((x[0]-center[0])**2 +
                                                (x[2]-center[2])**2) < radius
                                 )

class RandomSphere(RandomGenerator):
    def __init__(self, center, radius):
        assert radius > 0
        domain = [[center[0]-radius, center[0]+radius],
                  [center[1]-radius, center[1]+radius],
		          [center[2]-radius, center[2]+radius]]
        RandomGenerator.__init__(self, domain,
                                 lambda x: sqrt((x[0]-center[0])**2 +
                                                (x[1]-center[1])**2 +
						                        (x[2]-center[2])**2) < radius
                                 )

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    r_rectangle = RandomRectangle(0, 2, 1, 5).generate([100, 100],
                                                       method='tensor')
    r_circle = RandomCircle([0, 0], 1).generate([100, 100])

    for points in [r_rectangle, r_circle]:
        plt.figure()
        plt.scatter(points[:, 0], points[:, 1])
        plt.axis('equal')

    plt.show()
