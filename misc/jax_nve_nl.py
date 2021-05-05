from statistics import mean
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from jax import jit, lax, random
from jax_md import energy, space, simulate, quantity
import time
import asax
from asax.jax_utils import *
from jax.config import config
config.update("jax_enable_x64", True)


def initialize_system():
    atoms = bulk("Ar", cubic=True) * [5, 5, 5]
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    Stationary(atoms)

    # TODO: Remove later
    atoms.calc = asax.lj.LennardJones(epsilon=1.5, sigma=2.0, rc=10.0, ro=6.0)

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


def get_step_fn(apply_fn, neighbor_fn):
    def step_fn(i, state):
        state, neighbors = state
        neighbors = neighbor_fn(state.position, neighbors)
        state = apply_fn(state, neighbor=neighbors)
        return state, neighbors

    return step_fn


def run_nve(steps: int, step_fn, neighbor_fn, state, neighbors):
    i = 0

    step_times = []
    # potential_energy = []
    # kinetic_energy = []

    # print_every = 1
    old_time = time.monotonic()
    print('Step\tKE\tPE\tTotal Energy\ttime/step')
    print('----------------------------------------')

    while i < steps:
        # TODO: why 0 to 100?
        state, neighbors = lax.fori_loop(0, 100, step_fn, (state, neighbors))

        if neighbors.did_buffer_overflow:
            print("NL overflow, recomputing...")
            neighbors = neighbor_fn(state.position)
            continue

        # potential_energy += [energy_fn(state.position, neighbor=neighbors)]
        # kinetic_energy += [quantity.kinetic_energy(state.velocity)]

        now = time.monotonic()
        step_ms = round((now - old_time) * 1000, 2)
        step_times.append(step_ms)
        old_time = now
        # print("step {} took {} ms".format(i, step_ms))

        # if i % print_every == 0 and i > 0:
            # new_time = time.time()
            # print('{}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.2f}'.format(
            #     i * print_every, kinetic_energy[-1], potential_energy[-1], kinetic_energy[-1] + potential_energy[-1],
            #     (new_time - old_time) / print_every / 10.0))
            # old_time = new_time

        i += 1

    return step_times


def setup_nve(box, R):
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    displacement_fn = jit(displacement_fn)
    neighbor_fn, energy_fn = get_potential(displacement_fn, box)
    initial_neighbor_list = neighbor_fn(R)

    energy_fn = jit(energy_fn)
    shift_fn = jit(shift_fn)

    _, apply_fn = simulate.nve(energy_fn, shift_fn, dt=dt)
    initial_state = get_initial_nve_state(atoms)
    step_fn = get_step_fn(apply_fn, neighbor_fn)

    return initial_state, step_fn, initial_neighbor_list, neighbor_fn


steps = 1000
dt = 5 * 1e-15    # 5 fs
atoms, box, R = initialize_system()
start = time.monotonic()

initial_state, step_fn, initial_neighbor_list, neighbor_fn = setup_nve(box, R)
step_timings = run_nve(steps, step_fn, neighbor_fn, initial_state, initial_neighbor_list)

mean_step_time_ms = round(mean(step_timings[1:]), 2)
total_runtime_seconds = round(time.monotonic() - start, 2)

print("{} steps".format(steps))
print("{} seconds total".format(total_runtime_seconds))
print("{} ms/step (w/o first run)".format(mean_step_time_ms))
