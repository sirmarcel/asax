import time
from typing import List, Tuple, Dict

import numpy as np
from ase import units, Atoms
from jax import jit, lax
from jax.config import config
from jax_md import simulate, space, energy
from jax_md.energy import NeighborFn, NeighborList
from jax_md.simulate import ApplyFn

from asax import jax_utils
from asax.jax_utils import DisplacementFn, ShiftFn, EnergyFn, NVEState

config.update("jax_enable_x64", True)


class NveSimulation:
    displacement_fn: DisplacementFn
    shift_fn: ShiftFn
    energy_fn: EnergyFn
    neighbor_fn: NeighborFn
    apply_fn: ApplyFn

    initial_state: NVEState
    initial_neighbor_list: NeighborList

    initialization_time_ms: float
    # nl_recalculation_events: Dict[int, float]   # step, float
    nl_recalculation_events: List[int]
    # batch_times_ms: List[float]
    batch_times_ms: Dict[int, float]  # step, float

    def __init__(self, atoms: Atoms, dt: float, batch_size: int):
        self.atoms = atoms
        self.dt = dt
        self.batch_size = batch_size
        self.box = atoms.get_cell().array
        self.R = atoms.get_positions()
        self._initialize()

    def _initialize(self):
        start = time.monotonic()
        self.displacement_fn, self.shift_fn = self._setup_space()
        self.neighbor_fn, self.energy_fn = self._setup_potential(self.displacement_fn)
        self.initial_neighbor_list = self.neighbor_fn(self.R)
        self.initial_state, self.apply_fn = self._setup_nve(self.energy_fn, self.shift_fn)
        _, self.initialization_time_ms = self._measure_elapsed_time(start)

    def _measure_elapsed_time(self, start: float) -> Tuple[float, float]:
        """Returns current timestamp and elapsed time until then"""
        now = time.monotonic()
        return now, round((now - start) * 1000, 2)

    def _setup_space(self) -> Tuple[DisplacementFn, ShiftFn]:
        displacement_fn, shift_fn = space.periodic_general(self.box, fractional_coordinates=False)
        return jit(displacement_fn), jit(shift_fn)

    def _setup_potential(self, displacement_fn: DisplacementFn) -> Tuple[NeighborFn, EnergyFn]:
        sigma = 2.0
        epsilon = 1.5
        rc = 10.0
        ro = 6.0
        normalized_ro = ro / sigma
        normalized_rc = rc / sigma
        neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(displacement_fn, self.box,
                                                                    sigma=sigma,
                                                                    epsilon=epsilon,
                                                                    r_onset=normalized_ro,
                                                                    r_cutoff=normalized_rc)

        return neighbor_fn, jit(energy_fn)

    def _setup_nve(self, energy_fn: EnergyFn, shift_fn: ShiftFn) -> Tuple[NVEState, ApplyFn]:
        _, apply_fn = simulate.nve(energy_fn, shift_fn, dt=self.dt)
        return self._get_initial_nve_state(), apply_fn

    def _get_initial_nve_state(self) -> NVEState:
        R = self.atoms.get_positions()
        V = self.atoms.get_velocities()
        forces = self.atoms.get_forces()
        masses = self.atoms.get_masses()[0]
        return NVEState(R, V, forces, masses)

    def _step_fn(self, i, state):
        state, neighbors = state
        neighbors = self.neighbor_fn(state.position, neighbors)
        state = self.apply_fn(state, neighbor=neighbors)
        return state, neighbors

    def _get_step_fn(self, neighbor_fn: NeighborFn, apply_fn: ApplyFn):
        def step_fn(i, state):
            state, neighbors = state
            neighbors = neighbor_fn(state.position, neighbors)
            state = apply_fn(state, neighbor=neighbors)
            return state, neighbors

        return step_fn

    @property
    def step_times(self):
        """Returns average times per step within each batch (in miliseconds)"""
        return np.array(list(self.batch_times_ms.values())) / self.batch_size

    @property
    def total_simulation_time(self):
        """Returns the total simulation time (in seconds)"""
        return round(np.sum(self.step_times) * self.batch_size / 1000, 2)

    def run(self, steps: int, verbose=False):
        if steps % self.batch_size != 0:
            raise ValueError("Number of steps need to be dividable by batch_size")

        # self.batch_times_ms = []
        self.nl_recalculation_events = []
        self.batch_times_ms = {}
        # self.nl_recalculation_events = {}

        step_fn = self._get_step_fn(self.neighbor_fn, self.apply_fn)
        state = self.initial_state
        neighbors = self.initial_neighbor_list

        i = 0
        start = time.monotonic()

        while i < steps:
            i += self.batch_size
            state, neighbors = lax.fori_loop(0, self.batch_size, step_fn, (state, neighbors))

            if neighbors.did_buffer_overflow:
                neighbors = self.neighbor_fn(state.position)
                self.nl_recalculation_events.append(i)
                if verbose:
                    print("Steps {}/{}: Neighbor list overflow, recomputing...".format(i, steps))
                continue

            now, step_ms = self._measure_elapsed_time(start)
            # self.batch_times_ms.append(step_ms)
            self.batch_times_ms[i + self.batch_size] = step_ms
            start = now

            if verbose:
                print("Steps {}/{} took {} ms".format(i, steps, step_ms))


atoms = jax_utils.initialize_cubic_argon(multiplier=10)
sim = NveSimulation(atoms, dt=5 * units.fs, batch_size=5)
sim.run(steps=1000)

print("Total simulation time: {} s".format(sim.total_simulation_time))
print("Average runtime/step: {} ms".format(np.mean(sim.step_times)))

