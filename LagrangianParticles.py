# __authors__ = ('Mikael Mortensen <mikaem@math.uio.no>',
#                'Miroslav Kuchta <mirok@math.uio.no>')
# __date__ = '2014-19-11'
# __copyright__ = 'Copyright (C) 2011' + __authors__
# __license__  = 'GNU Lesser GPL version 3 or any later version'
'''
This module contains functionality for Lagrangian tracking of particles with
DOLFIN
'''

import dolfin as df
import numpy as np
import copy
import ufc
from mpi4py import MPI as pyMPI
from collections import defaultdict
import matplotlib.colors as colors
import matplotlib.cm as cmx
import time
import random
from particle_generators import RandomCircle, RandomSphere
from mpl_toolkits.mplot3d import Axes3D


__DEBUG__ = False

# Disable printing
__DEBUG__ = False

comm = pyMPI.COMM_WORLD


# collisions tests return this value or -1 if there is no collision
__UINT32_MAX__ = np.iinfo('uint32').max

class Particle:
    'Lagrangian particle with position and some other passive properties.'
    def __init__(self, x):
        self.position = x
        self.properties = {}
        self.properties["w"] = 1.0
        

    def send(self, dest):
        'Send particle to dest.'
        comm.Send(self.position, dest=dest)
        comm.send(self.properties, dest=dest)

    def recv(self, source):
        'Receive info of a new particle sent from source.'
        comm.Recv(self.position, source=source)
        self.properties = comm.recv(source=source)


class CellWithParticles(df.Cell):
    'Dolfin cell with list of particles that it contains.'
    def __init__(self, mesh, cell_id, particle):
        # Initialize parent -- create Cell with id on mesh
        df.Cell.__init__(self, mesh, cell_id)
        # Make an empty list of particles that I carry
        self.particles = []
        self += particle

    def __add__(self, particle):
        'Add single particle to cell.'
        assert isinstance(particle, (Particle, np.ndarray))
        if isinstance(particle, Particle):
            self.particles.append(particle)
            return self
        else:
            return self.__add__(Particle(particle))

    def __len__(self):
        'Number of particles in cell.'
        return len(self.particles)
    
    def total_weight(self):
        if len(self) == 0:
            return 0.0
        
        else:
            w = 0
            for particle in self.particles:
                w += particle.properties["w"]
                #print particle.properties["w"]
            return w

    def average_weight(self):
        if len(self) == 0:
            a = 0.0
        else:
            a = self.total_weight()/float(len(self))
        return a
    
    def set_concentration(self, concentration_factor):
        for p in self.particles:
            p.properties["w"] = concentration_factor(p.position)

    def set_weight(self, w):
        for p in self.particles:
            p.properties["w"] = w

    def mult_weight(self, factor):
        for p in self.particles:
            p.properties["w"] = p.properties["w"]*factor
        
class ParticleSource():
    '''
    Fills all cells within a domain up with particles such that the density is constant over the area.
    If N particles leave one cell, we place N particles on a random position in the cell, so the density remains constant.
    The number of particles we wish to have in one cell depends on its volume. The number of particles in each cell
    is: mean density * cell.volume()
    '''
    def __init__(self, particles_per_cell, subdomain, mesh, lp, consentration_factor=lambda x: 1.0, random_generator=None):
        self.lp = lp
        self.particles_per_cell = particles_per_cell
        self.subdomain = subdomain
        self.mesh = mesh
        self.cells = self.find_cell_ids()
        self.mean_volume = self.find_mean_volume()
        self.concentration_factor = consentration_factor
        self.random_generator = random_generator

    def num_global_cells(self):
        n = len(self.cells)
        n = comm.allgather(n)
        return sum(n)
        
        
    def find_cell_ids(self):
        '''return list of cell ids of cells within the domain'''
        mf = df.MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
        mf.set_all(0)
        self.subdomain.mark(mf, 1)
        marked_cells = df.SubsetIterator(mf, 1)
        cells = []
        for cell in marked_cells:   
            cells.append(cell.index())
            
        # df.plot(mf, interactive=True)
                
        return cells
    
    def find_mean_volume(self):
        '''
        returns the mean volume over the entire domain
        '''
        cells = self.cells
        vol = []
        #print len(self.cells)
        for cell in cells:
            dolfin_cell = df.Cell(self.mesh, cell)
            vol.append(dolfin_cell.volume())
        mean = np.mean(vol) if len(vol) > 0 else 0.0
        mean = comm.allreduce(mean)
        mean = np.mean(mean)
            
        return mean

    def particles_in_domain(self):
        cells = self.cells
        count = 0
        for cell in cells:
            if cell in self.lp.particle_map.keys():
                count += len(self.lp.particle_map[cell])

        count = comm.allgather(count)
        return sum(count)
    
    def pick_point(self, cell_id, dim):
        '''
        Picks random point inside a triangle or tetrahetron.
        Tetrahetron picking is based on the paper http://vcg.isti.cnr.it/publications/papers/rndtetra_a.pdf.
        Triangle picking is trivial.
        '''
        #df.plot(global_mesh, interactive=True)
        cell = df.Cell(self.mesh, cell_id)
        vertices = cell.get_vertex_coordinates()
        if dim == 2:
            v0 = [vertices[0], vertices[1]]
            v1 = [vertices[2], vertices[3]]
            v2 = [vertices[4], vertices[5]]
        elif dim == 3:
            v0 = [vertices[0], vertices[1], vertices[2]]
            v1 = [vertices[3], vertices[4], vertices[5]]
            v2 = [vertices[6], vertices[7], vertices[8]]
            v3 = [vertices[9], vertices[10], vertices[11]]
        
        if dim == 3:
            s = random.uniform(0,1)
            t = random.uniform(0,1)
            u = random.uniform(0,1)
            
            # Fold cube into prism
            if (s + t > 1.0):
                s = 1. - s
                t = 1. - t
            
            # Fold prism into tetrahetron
            if (t + u > 1.0):
                tmp = u
                u = 1. - s - t
                t = 1. - tmp
                
                
            elif (s + t + u > 1.0):
                tmp = u
                u = s + t + u - 1.0
                s = 1. - t - tmp
            
            a = 1. - s - t - u
            point = np.array(v0)*a + np.array(v1)*s + np.array(v2)*t + np.array(v3)*u
            return [point[0], point[1], point[2]]
        elif dim == 2:
            b0 = random.uniform(0,1)
            b1 = ( 1.0 - b0 ) * random.uniform(0,1);
            b2 = 1.0 - b0 - b1;

            point = np.array(v0) * b0 + np.array(v1) * b1 + np.array(v2) * b2;
            #print cell.contains(df.Point(point[0], point[1]))
            return [point[0], point[1]]
                
    
    def select_random_points(self, cell_id, num):
        '''
        Selects random points for particles to be added
        '''
        points = []
        for i in range(num):
            #print i
            rand_point = self.pick_point(cell_id, self.mesh.topology().dim())
            points += [rand_point]
        
        #print "len points:",len(points)
        return points
    
    def select_random_particles(self, cell_id, num):
	    '''
	    Selects random particles already existing
	    '''
	    all_particles = self.lp.particle_map[cell_id].particles
	    rand_particles = random.sample(range(len(all_particles)), num)
	    return rand_particles

    def midpoint(self, cell_id):
        cell = df.Cell(self.mesh, cell_id)
        midpoint = cell.midpoint()
        x = []
        for i in range(self.mesh.topology().dim()):
            x.append(midpoint[i])
        return x

        
    def apply_source(self):
        '''
        Adds particles to the cells that have less particles than it should.
	    Removes particles if too many
        '''
        particles_to_be_added = list()
        properties_d = dict(w=list())
        #particles_to_be_removed = dict()
        #particles_to_be_removed_local = {}
        num_cells = len(self.cells)
        
        if num_cells > 0:
            mean_density = self.particles_per_cell/self.mean_volume
        else:
            mean_density = 0.0
        
        cells = self.cells

        for index, cell_id in enumerate(cells):
            if cell_id in self.lp.particle_map.keys():
                cwp = self.lp.particle_map[cell_id]
                midpoint = self.midpoint(cell_id)
                num_particles = int(round(cwp.volume()*mean_density))
                if len(cwp) < num_particles:
                    # Adds particles if too few
                    random_particles = self.select_random_points(cell_id, num_particles - len(cwp))
                    particles_to_be_added += random_particles
                    for p in random_particles:
                        properties_d["w"].append(self.concentration_factor(p))
               # elif len(cwp) > num_particles:
		       #     # Removes particles if too many
		       #     particles_to_be_removed_local.update(cell_id=self.select_random_particles(cell_id, len(cwp) - num_particles))
                    
            else:
                midpoint = self.midpoint(cell_id)
                cell = df.Cell(self.mesh, cell_id)
                num_particles = int(round(cell.volume()*mean_density))
                random_particles = self.select_random_points(cell_id, num_particles)
                particles_to_be_added += random_particles
                for p in random_particles:
                        properties_d["w"].append(self.concentration_factor(p))
        
	    # Must be same on all procecess    
        particles_to_be_added = comm.allreduce(particles_to_be_added)
        #particles_to_be_removed_gather = comm.allgather(particles_to_be_removed_local)
	    
        #for i, particles in enumerate(particles_to_be_removed_gather):
	    #    particles_to_be_removed.update(particles)
        #print properties_d
        self.lp.add_particles(np.array(particles_to_be_added), properties_d)
        #self.lp.remove_particles(particles_to_be_removed)

    def apply_weight(self):
        cells = self.cells
        for index, cell_id in enumerate(cells):
            if cell_id in self.lp.particle_map.keys():
                cwp = self.lp.particle_map[cell_id]
                midpoint = self.midpoint(cell_id)
                cwp.set_concentration(self.concentration_factor)


    def apply_source_all(self, n_to_be_added):
        'Adds n_to_be_added particles in random positions in the domain'
        properties_d = dict(w=list())
        #n_to_be_added = (self.particles_per_cell*num_cells - num_particles) / (4./3 * np.pi * 0.5**3) 
        N = np.zeros(self.mesh.topology().dim())
        N[:] = np.power(n_to_be_added, 1./self.mesh.topology().dim())
        particles_to_be_added = self.random_generator.generate(N, method="full") 
        for p in particles_to_be_added:
            properties_d["w"].append(self.concentration_factor(p))
        self.lp.add_particles(particles_to_be_added, properties_d)




class CellParticleMap(dict):
    'Dictionary of cells with particles.'
    def __add__(self, ins):
        '''
        Add ins to map:
            ins is either (mesh, cell_id, particle) or
                          (mesh, cell_id, particle, particle_properties)
        '''
        assert isinstance(ins, tuple) and len(ins) in (3, 4)
        # If the cell_id is in map add the particle
        if ins[1] in self:
            self[ins[1]] += ins[2]
        # Other wise create new cell
        else:
            self[ins[1]] = CellWithParticles(ins[0], ins[1], ins[2])
        # With particle_properties, update properties of the last added particle
        if len(ins) == 4:
            self[ins[1]].particles[-1].properties.update(ins[3])

        return self

    def pop(self, cell_id, i):
        'Remove i-th particle from the list of particles in cell with cell_id.'
        # Note that we don't check for cell_id being a key or cell containg
        # at least i particles.
        particle = self[cell_id].particles.pop(i)

        # If the cell is empty remove it from map
        if len(self[cell_id]) == 0:
            del self[cell_id]

        return particle

    def total_number_of_particles(self):
        'Total number of particles in all cells of the map.'
        return sum(map(len, self.itervalues()))


class LagrangianParticles:
    'Particles moved by the velocity field in V.'
    def __init__(self, V):
        self.__debug = __DEBUG__

        self.V = V
        self.mesh = V.mesh()
        self.mesh.init(2, 2)  # Cell-cell connectivity for neighbors of cell
        self.tree = self.mesh.bounding_box_tree()  # Tree for isection comput.
        self.DG0 = df.FunctionSpace(self.mesh, "DG", 0)
        self.CG1 = df.FunctionSpace(self.mesh, "CG", 1)
        self.rho = df.Function(self.DG0)
        self.maxrho = 1.0
        self.dt = 0.001
        self.h = 0.01
        self.characteristic_density = None

        # Allocate some variables used to look up the velocity
        # Velocity is computed as U_i*basis_i where i is the dimension of
        # element function space, U are coefficients and basis_i are element
        # function space basis functions. For interpolation in cell it is
        # advantageous to compute the resctriction once for cell and only
        # update basis_i(x) depending on x, i.e. particle where we make
        # interpolation. This updaea mounts to computing the basis matrix
        self.ufc_cell = ufc.cell()
        self.dim = self.mesh.topology().dim()

        self.element = V.dolfin_element()
        self.num_tensor_entries = 1
        for i in range(self.element.value_rank()):
            self.num_tensor_entries *= self.element.value_dimension(i)
        # For VectorFunctionSpace CG1 this is 3
        self.coefficients = np.zeros(self.element.space_dimension())
        self.coefficients_p = np.zeros(self.element.space_dimension())
        self.coefficients_pp = np.zeros(self.element.space_dimension())
        # For VectorFunctionSpace CG1 this is 3x3
        self.basis_matrix = np.zeros((self.element.space_dimension(),
                                      self.num_tensor_entries))
        self.basis_matrix_p_k1 = np.zeros((self.element.space_dimension(),
                                      self.num_tensor_entries))
        self.basis_matrix_p_k2 = np.zeros((self.element.space_dimension(),
                                      self.num_tensor_entries))
        
        self.basis_matrix_k3 = np.zeros((self.element.space_dimension(),
                                      self.num_tensor_entries))
        self.basis_matrix_pp = np.zeros((self.element.space_dimension(),
                                      self.num_tensor_entries))
        

        # Allocate a dictionary to hold all particles
        self.particle_map = CellParticleMap()

        # Allocate some MPI stuff
        self.num_processes = comm.Get_size()
        self.myrank = comm.Get_rank()
        self.all_processes = range(self.num_processes)
        self.other_processes = range(self.num_processes)
        self.other_processes.remove(self.myrank)
        self.my_escaped_particles = np.zeros(1, dtype='I')
        self.tot_escaped_particles = np.zeros(self.num_processes, dtype='I')
        # Dummy particle for receiving/sending at [0, 0, ...]
        self.particle0 = Particle(np.zeros(self.mesh.geometry().dim()))

    def add_particles(self, list_of_particles, properties_d=None):
        '''Add particles and search for their home on all processors.
           Note that list_of_particles must be same on all processes. Further
           every len(properties[property]) must equal len(list_of_particles).
        '''
        if properties_d is not None:
            n = len(list_of_particles)
            assert all(len(sub_list) == n
                       for sub_list in properties_d.itervalues())
            # Dictionary that will be used to feed properties of single
            # particles
            properties = properties_d.keys()
            particle_properties = dict((key, 0) for key in properties)

            has_properties = True
        else:
            has_properties = False

        pmap = self.particle_map
        my_found = np.zeros(len(list_of_particles), 'I')
        all_found = np.zeros(len(list_of_particles), 'I')
        for i, particle in enumerate(list_of_particles):
            c = self.locate(particle)
            #print c
            if not (c == -1 or c == __UINT32_MAX__):
                my_found[i] = True
                if not has_properties:
                    pmap += self.mesh, c, particle
                else:
                    # Get values of properties for this particle
                    for key in properties:
                        particle_properties[key] = properties_d[key][i]
                    pmap += self.mesh, c, particle, particle_properties
        # All particles must be found on some process
        comm.Reduce(my_found, all_found, root=0)

        if self.myrank == 0:
            missing = np.where(all_found == 0)[0]
            n_missing = len(missing)

            #assert n_missing == 0,\
            #    '%d particles are not located in mesh' % n_missing

            # Print particle info
            if self.__debug:
                for i in missing:
                    print 'Missing', list_of_particles[i].position

                n_duplicit = len(np.where(all_found > 1)[0])
                print 'There are %d duplicit particles' % n_duplicit
    
    def remove_particles(self, list_of_particles):
	'Remove particles from home processors'
	pmap = self.particle_map
	
	for cell_id, particle in list_of_particles.iteritems():
	    if cell_id in pmap.keys():
		pmap.pop(cell_id, particle)
	

    def step(self, u, dt):
        'Move particles by forward Euler x += u*dt'
        start = time.time()
	
        for cwp in self.particle_map.itervalues():
            # Restrict once per cell
            u.restrict(self.coefficients,
                       self.element,
                       cwp,
                       cwp.get_vertex_coordinates(),
                       self.ufc_cell)
            for particle in cwp.particles:
                x = particle.position
                # Compute velocity at position x
                self.element.evaluate_basis_all(self.basis_matrix,
                                                x,
                                                cwp.get_vertex_coordinates(),
                                                cwp.orientation())
                x[:] = x[:] + dt*np.dot(self.coefficients, self.basis_matrix)[:]
        
        # Recompute the map
        stop_shift = time.time() - start
        start = time.time()
        info = self.relocate()
        stop_reloc = time.time() - start
        # We return computation time per process
        return (stop_shift, stop_reloc)

    def average_weight(self):
        w = 0.0
        for cwp in self.particle_map.itervalues():
            w += cwp.average_weight()
        w = w/float(len(self.particle_map))

        w = comm.allgather(w)
        w = np.average(w)
        return w


    def weight(self):
        num_particles = self.total_number_of_particles()

        num_cells = len(self.particle_map)
        num_cells = comm.allgather(num_cells)
        num_cells = sum(num_cells)

        if len(self.particle_map) > 0:
            for cwp in self.particle_map.itervalues():
                cwp.set_weight(self.average_weight() / cwp.average_weight())
    

    def relocate(self):
        # Relocate particles on cells and processors
        p_map = self.particle_map
        # Map such that map[old_cell] = [(new_cell, particle_id), ...]
        # Ie new destination of particles formerly in old_cell
        new_cell_map = defaultdict(list)
        for cwp in p_map.itervalues():
            for i, particle in enumerate(cwp.particles):
                point = df.Point(*particle.position)
                # Search only if particle moved outside original cell
                #print "cwp %g contains %g" % (cwp.index(), cwp.contains(point))
                if not cwp.contains(point):
                    #print "Moved outside"
                    found = False
                    # Check neighbor cells
                    for neighbor in df.cells(cwp):
                        if neighbor.contains(point):
                            new_cell_id = neighbor.index()
                            #print "New cell id:",new_cell_id
                            found = True
                            break
                    # Do a completely new search if not found by now
                    if not found:
                        new_cell_id = self.locate(particle)
                        #print new_cell_id
                    # Record to map
                    new_cell_map[cwp.index()].append((new_cell_id, i))
	

        # Rebuild locally the particles that end up on the process. Some
        # have cell_id == -1, i.e. they are on other process
        list_of_escaped_particles = []
        for old_cell_id, new_data in new_cell_map.iteritems():
            # We iterate in reverse becasue normal order would remove some
            # particle this shifts the whole list!
            for (new_cell_id, i) in sorted(new_data,
                                           key=lambda t: t[1],
                                           reverse=True):
                particle = p_map.pop(old_cell_id, i)

                if new_cell_id == -1 or new_cell_id == __UINT32_MAX__ :
                    list_of_escaped_particles.append(particle)
                else:
                    p_map += self.mesh, new_cell_id, particle
	

        # Create a list of how many particles escapes from each processor
        self.my_escaped_particles[0] = len(list_of_escaped_particles)
        # Make all processes aware of the number of escapees
        comm.Allgather(self.my_escaped_particles, self.tot_escaped_particles)
	
        if comm.Get_rank() is not 0:
            for particle in list_of_escaped_particles:
                particle.send(0)
	
        # Receive the particles escaping from other processors
        if self.myrank == 0:
            for proc in self.other_processes:
                for i in range(self.tot_escaped_particles[proc]):
                    self.particle0.recv(proc)
                    list_of_escaped_particles.append(copy.deepcopy(self.particle0))

        # Put all travelling particles on all processes, then perform new search
        travelling_particles = comm.bcast(list_of_escaped_particles, root=0)
        self.add_particles(travelling_particles)

    def total_number_of_particles(self):
        'Return number of particles in total and on process.'
        num_p = self.particle_map.total_number_of_particles()
        tot_p = comm.allreduce(num_p)
        return (tot_p, num_p)

    def locate(self, particle):
        'Find mesh cell that contains particle.'
        assert isinstance(particle, (Particle, np.ndarray))
        if isinstance(particle, Particle):
            # Convert particle to point
            point = df.Point(*particle.position)
            #print self.tree.compute_first_entity_collision(point)
            return self.tree.compute_first_entity_collision(point)
        else:
            return self.locate(Particle(particle))

    def scatter(self, fig, dim=2, skip=1):
        'Scatter plot of all particles on process 0'
        ax = fig.gca()
        if dim==3:
            ax = fig.add_subplot(111, projection='3d')

        p_map = self.particle_map
        all_particles = np.zeros(self.num_processes, dtype='I')
        my_particles = p_map.total_number_of_particles()
        # Root learns about count of particles on all processes
        comm.Gather(np.array([my_particles], 'I'), all_particles, root=0)

        # Slaves should send to master
        if self.myrank > 0:
            for cwp in p_map.itervalues():
                for p in cwp.particles:
                    p.send(0)
        else:
            # Receive on master
            received = defaultdict(list)
            received[0] = [copy.copy(p.position)
                           for cwp in p_map.itervalues()
                           for p in cwp.particles]
            for proc in self.other_processes:
                # Receive all_particles[proc]
                for j in range(all_particles[proc]):
                    self.particle0.recv(proc)
                    received[proc].append(copy.copy(self.particle0.position))

            cmap = cmx.get_cmap('jet')
            cnorm = colors.Normalize(vmin=0, vmax=self.num_processes)
            scalarMap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

            for proc in received:
                # Plot only if there is something to plot
                particles = received[proc]
                if len(particles) > 0:
                    xy = np.array(particles)
                    if dim == 2:
                        ax.scatter(xy[::skip, 0], xy[::skip, 1],
                                    label='%d' % proc,
                                    c=scalarMap.to_rgba(proc),
                                    edgecolor='none', s=1.0)
                    elif dim == 3:
                        ax.scatter(xy[::skip, 0], xy[::skip, 1],xy[::skip, 2],
                                    label='%d' % proc,
                                    c=scalarMap.to_rgba(proc),
                                    edgecolor='none', s=2.0)

            ax.legend(loc='best')
            #ax.axis([0, 1, 0, 1])
    
    
    def bar(self, fig):
        'Bar plot of particle distribution.'
        ax = fig.gca()

        p_map = self.particle_map
        all_particles = np.zeros(self.num_processes, dtype='I')
        my_particles = p_map.total_number_of_particles()
        # Root learns about count of particles on all processes
        comm.Gather(np.array([my_particles], 'I'), all_particles, root=0)

        if self.myrank == 0 and self.num_processes > 1:
            ax.bar(np.array(self.all_processes)-0.25, all_particles, 0.5)
            ax.set_xlabel('proc')
            ax.set_ylabel('number of particles')
            ax.set_xlim(-0.25, max(self.all_processes)+0.25)
            return np.sum(all_particles)
        else:
            return None
        
    def particle_density(self, rho, normalize=0.0):
        'Make rho represent particle density.'
        assert rho.ufl_element().family() == 'Discontinuous Lagrange'
        assert rho.ufl_element().degree() == 0
        assert rho.ufl_element().value_shape() == ()

        vec = rho.vector()
        vec.zero()

        #self.weight()

        dofmap = rho.function_space().dofmap()
        first, last = dofmap.ownership_range()
        values = np.zeros(last-first)

        for cell_id, cwp in self.particle_map.iteritems():
            dof = dofmap.cell_dofs(cell_id)[0]
            if first <= first+dof < last:
                values[dof] = cwp.total_weight()/cwp.volume()

        vec.set_local(values)
        vec.apply('insert')



