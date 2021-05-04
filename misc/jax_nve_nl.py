from ase.build import bulk
import numpy as np
from jax import jit, lax, random
from jax_md import energy, space, simulate, quantity
import time
from jax.config import config

config.update("jax_enable_x64", True)


def initialize_system():
    atoms = bulk("Ar", cubic=True) * [5, 5, 5]
    atoms.set_cell(1.05 * atoms.get_cell(), scale_atoms=True)
    box = atoms.get_cell().array
    R = atoms.get_positions()
    return atoms, box, R


def get_potential(displacement_fn, box):
    sigma = 2.0
    epsilon = 1.5
    rc = 10.0
    ro = 6.0
    normalized_ro = ro / sigma
    normalized_rc = rc / sigma
    return energy.lennard_jones_neighbor_list(displacement_fn, box,
                                              sigma=sigma,
                                              epsilon=epsilon,
                                              r_onset=normalized_ro,
                                              r_cutoff=normalized_rc)


def step_fn(i, state):
    state, neighbors = state
    neighbors = neighbor_fn(state.position, neighbors)
    state = apply_fn(state, neighbor=neighbors)
    return state, neighbors


def run_nve(steps: int, state, neighbors):
    i = 0

    potential_energy = []
    kinetic_energy = []

    print_every = 1
    old_time = time.time()
    print('Step\tKE\tPE\tTotal Energy\ttime/step')
    print('----------------------------------------')

    while i < steps:
        # TODO: why 0 to 100?
        state, neighbors = lax.fori_loop(0, 100, step_fn, (state, neighbors))

        if neighbors.did_buffer_overflow:
            print("NL overflow, recomputing...")
            neighbors = neighbor_fn(state.position)
            continue

        potential_energy += [energy_fn(state.position, neighbor=neighbors)]
        kinetic_energy += [quantity.kinetic_energy(state.velocity)]

        if i % print_every == 0 and i > 0:
            new_time = time.time()
            print('{}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.2f}'.format(
                i * print_every, kinetic_energy[-1], potential_energy[-1], kinetic_energy[-1] + potential_energy[-1],
                (new_time - old_time) / print_every / 10.0))
            old_time = new_time

        i += 1


steps = 1000
time_step = None
kT = None
start = time.monotonic()

key = random.PRNGKey(0)
atoms, box, R = initialize_system()
displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
neighbor_fn, energy_fn = get_potential(displacement_fn, box)
energy_fn = jit(energy_fn)

# build initial neighbor list and state for NVE
neighbors = neighbor_fn(R)
init_fn, apply_fn = simulate.nve(energy_fn, shift_fn, dt=1e-2)
state = init_fn(key, R, kT=0.0, neighbor=neighbors)

# run NVE
run_nve(steps, state, neighbors)
elapsed_seconds = round(time.monotonic() - start, 2)
mean_step_time_ms = round((elapsed_seconds / steps) * 1000, 2)

print("{} steps".format(steps))
print("{} seconds total".format(elapsed_seconds))
print("{} ms/step".format(mean_step_time_ms))
