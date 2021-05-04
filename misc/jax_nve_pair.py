from ase.atoms import Atoms
from ase.build import bulk
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, random
from jax.config import config
config.update("jax_enable_x64", True)
from jax_md import energy, space, simulate, quantity
import jax_utils
import time

def generate_system(key, N: int, box_size: float):
    return random.uniform(key, (N, 3), minval=0.0, maxval=box_size, dtype=np.float64)

def initialize_system():
    atoms = bulk("Ar", cubic=True) * [5, 5, 5]
    atoms.set_cell(1.05 * atoms.get_cell(), scale_atoms=True)
    box = atoms.get_cell().array
    R = atoms.get_positions()
    return atoms, box, R

def get_potential(displacement_fn):
    sigma = 2.0
    epsilon = 1.5
    rc = 10.0
    ro = 6.0
    normalized_ro = ro / sigma
    normalized_rc = rc / sigma
    return energy.lennard_jones_pair(displacement_fn, 
                                        sigma=sigma,
                                        epsilon=epsilon,
                                        r_onset=normalized_ro,
                                        r_cutoff=normalized_rc)

key = random.PRNGKey(0)
# N = 500
# box_size = 120.0
# R = generate_system(key, N, box_size)

atoms, box, R = initialize_system()
displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
energy_fn = get_potential(displacement_fn)

init_fn, apply_fn = simulate.nve(energy_fn, shift_fn, 1e-2)
step_fn = jit(lambda i, state: apply_fn(state))
state = init_fn(key, R, kT=0.0)

i = 0
max_steps = 100

potential_energy = []
kinetic_energy = []

print_every = 1
old_time = time.time()
print('Step\tKE\tPE\tTotal Energy\ttime/step')
print('----------------------------------------')

while i < max_steps:

    # why 0 to 100?
    state = lax.fori_loop(0, 100, step_fn, state)
    potential_energy += [energy_fn(state.position)]
    kinetic_energy += [quantity.kinetic_energy(state.velocity)]

    if i % print_every == 0 and i > 0:
        new_time = time.time()
        print('{}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.2f}'.format(
            i * print_every, kinetic_energy[-1], potential_energy[-1], kinetic_energy[-1] + potential_energy[-1], 
            (new_time - old_time) / print_every / 10.0))
        old_time = new_time

    i += 1

