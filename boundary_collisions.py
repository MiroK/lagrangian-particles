
def remove_particles(lp, predicates):
    '''Remove Lagrangian particles whose positions eval True for some predicate'''
    pmap = lp.particle_map

    nparticles = lp.total_number_of_particles()
    if lp.myrank == 0:
        print(f'Preremoval {nparticles}')

    cell_indices = tuple(pmap.keys())
    for cell_id in cell_indices:
        # Collect indices of particles to be removed
        remove = [i
                  for i, p in enumerate(pmap[cell_id].particles)
                  if any(predicate(p.position) for predicate in predicates)]
        # Do it
        [pmap.pop(cell_id, i) for i in sorted(remove, reverse=True)]

    nparticles = lp.total_number_of_particles()
    if lp.myrank == 0:
        print(f'Postremoval {nparticles}')    

    return lp
