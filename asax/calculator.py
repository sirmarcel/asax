from abc import ABC, abstractmethod
from typing import Dict, Tuple, Callable

import numpy as np
from ase.stress import full_3x3_to_voigt_6_stress
from jax import jit
import jax.numpy as jnp
from jax.config import config
from jax_md import space, energy, partition
from ase.atoms import Atoms
from ase.calculators.abc import GetPropertiesMixin
from ase.calculators.calculator import compare_atoms, PropertyNotImplementedError
from jax_md.energy import NeighborFn

from asax import jax_utils
from asax.jax_utils import EnergyFn, PotentialFn


class Calculator(GetPropertiesMixin, ABC):
    implemented_properties = ["energy", "forces"]

    displacement: space.DisplacementFn
    shift: space.ShiftFn
    potential: jax_utils.PotentialFn

    neighbor_fn: energy.NeighborFn
    neighbors: partition.NeighborList

    def __init__(self, x64=True, stress=False):
        self.x64 = x64
        config.update("jax_enable_x64", self.x64)

        self.atoms: Atoms = None
        self.results = {}
        self.stress = stress

    @property
    def R(self) -> jnp.array:
        return jnp.float64(self.atoms.get_positions())

    @property
    def box(self) -> jnp.array:
        # box as vanilla np.array causes strange indexing errors with neighbor lists now and
        return jnp.float64(self.atoms.get_cell().array)

    def update(self, atoms: Atoms):
        if atoms is None and self.atoms is None:
            raise RuntimeError("Need an Atoms object to do anything!")

        changes = compare_atoms(self.atoms, atoms)

        if changes:
            self.results = {}
            self.atoms = atoms.copy()

            if self.need_setup(changes):
                self.setup()

    def need_setup(self, changes):
        return "cell" in changes or "pbc" in changes or "numbers" in changes

    def setup(self):
        """Create displacement, shift and potential"""
        self.displacement, self.shift = self.get_displacement(self.atoms)
        self.potential = self.get_potential()

    def get_displacement(self, atoms: Atoms):
        if not all(atoms.get_pbc()):
            return space.free()

        return space.periodic_general(self.box, fractional_coordinates=False)

    @abstractmethod
    def get_energy_function(self) -> Tuple[NeighborFn, EnergyFn]:
        pass

    def get_potential(self) -> PotentialFn:
        self.neighbor_fn, energy_fn = self.get_energy_function()
        self.update_neighbor_list()

        if self.stress:
            return jit(
                jax_utils.strained_neighbor_list_potential(
                    energy_fn, self.neighbors, self.box
                )
            )

        return jit(
            jax_utils.unstrained_neighbor_list_potential(
                energy_fn, self.neighbors
            )
        )

    def update_neighbor_list(self):
        self.neighbors = self.neighbor_fn(self.R)

    def compute_properties(self) -> Dict:
        if self.neighbors.did_buffer_overflow:
            self.update_neighbor_list()

        properties = self.potential(self.R)
        (
            potential_energy,
            potential_energies,
            forces,
            stress,
        ) = jax_utils.block_and_dispatch(properties)

        result = {
            "energy": potential_energy,
            "energies": potential_energies,
            "forces": forces,
        }

        if stress is not None:
            result["stress"] = full_3x3_to_voigt_6_stress(stress)
        return result

    # ase plumbing

    def calculate(self, atoms=None, **kwargs):
        self.update(atoms)
        self.results = self.compute_properties()

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(f"{name} property not implemented")

        self.update(atoms)

        if name not in self.results:
            if not allow_calculation:
                return None
            self.calculate(atoms=atoms)

        if name not in self.results:
            # For some reason the calculator was not able to do what we want,
            # and that is OK.
            raise PropertyNotImplementedError(f"{name} property not present in results!")

        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def get_potential_energy(self, atoms=None):
        return self.get_property(name="energy", atoms=atoms)
