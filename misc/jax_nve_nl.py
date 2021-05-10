from statistics import mean
from typing import Type, List

from ase.build import bulk
from ase.calculators.lj import LennardJones as aLJ
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from jax import jit, lax, random
from jax_md import energy, space, simulate, quantity
import time

from jax_md.energy import NeighborFn, NeighborList
from jax_md.simulate import Simulator, ApplyFn

from asax.jax_utils import *
from jax.config import config
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
    step_times_ms: List[float]
    nl_recalculation_events: List[bool]

    potential_energy: List[float]
    kinetic_energy: List[float]

    def __init__(self, atoms: Atoms, dt: float):
        self.atoms = atoms
        self.dt = dt
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

    def run(self, steps: int):
        self.step_times_ms = []
        self.nl_recalculation_events = []
        self.potential_energy = []
        self.kinetic_energy = []

        step_fn = self._get_step_fn(self.neighbor_fn, self.apply_fn)
        state = self.initial_state
        neighbors = self.initial_neighbor_list

        i = 0
        start = time.monotonic()
        while i < steps:
            state, neighbors = lax.fori_loop(0, 5, step_fn, (state, neighbors))       # TODO: why 0 to 100?
            if neighbors.did_buffer_overflow:
                print("NL overflow, recomputing...")
                neighbors = self.neighbor_fn(state.position)
                self.nl_recalculation_events.append(i)
                continue

            # self.potential_energy += [self.energy_fn(state.position, neighbor=neighbors)]
            # self.kinetic_energy += [quantity.kinetic_energy(state.velocity)]

            now, step_ms = self._measure_elapsed_time(start)
            self.step_times_ms.append(step_ms)
            start = now
            i += 1


def initialize_cubic_argon():
    atoms = bulk("Ar", cubic=True) * [5, 5, 5]
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    Stationary(atoms)
    
    atoms.calc = aLJ(sigma=2.0, epsilon=1.5, rc=10.0, ro=6.0)  # TODO: Remove later
    return atoms


# atoms = initialize_system()
# sim = NveSimulation(atoms, dt=5 * 1e-15)    # 5 fs
# sim.run(steps=1000)

# print(sim.step_times_ms)
