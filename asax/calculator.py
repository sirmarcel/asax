from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from jax.config import config
from jax_md import space
from ase.atoms import Atoms
from ase.calculators.abc import GetPropertiesMixin
from ase.calculators.calculator import compare_atoms, PropertyNotImplementedError
from asax import utils, jax_utils


class Calculator(GetPropertiesMixin, ABC):

    # TODO: Can this be abstract?
    implemented_properties = ["energy", "forces"]
    displacement: space.DisplacementFn
    potential: jax_utils.PotentialFn

    def __init__(self, x64=True):
        self.x64 = x64
        config.update("jax_enable_x64", self.x64)

        self.atoms_cache: Atoms = None
        self.results = {}

    def update(self, atoms: Atoms):
        if atoms is None and self.atoms_cache is None:
            raise RuntimeError("Need an Atoms object to do anything!")

        if self.atoms_cache is None:
            self.atoms_cache = atoms.copy()
            self.results = {}
            self.on_atoms_changed()
            self.setup()
            return

        changes = compare_atoms(self.atoms_cache, atoms)
        if not changes:
            return

        # cache not empty and we got a new atom that has changes
        # => clear results, clear cache, re-run update function to write new atom to cache
        self.results = {}
        if "cell" in changes:
            # TODO: Does this include switches from bulk to molecules?
            # => displacement only requires re-initialization if this is the case
            self.atoms_cache = None
            self.update(atoms)
            return

        # TODO: Detect changes in atom count/shape
        # => potential only requires re-initialization if this is the case

        # there are changes, but not within the cell.
        # => clear results, but write directly to the cache without copying.
        # TODO: why this?
        self.atoms_cache = atoms
        self.on_atoms_changed()
        self.setup()


    @abstractmethod
    def on_atoms_changed(self):
        """Called whenever a new atoms object is passed so that child classes can react accordingly."""
        pass

    def setup(self):
        self.displacement = jax_utils.get_displacement(self.atoms_cache)
        self.potential = self.get_potential()

    @property
    def R(self):
        return self.atoms_cache.get_positions()

    @property
    def box(self):
        return self.atoms_cache.get_cell().array

    @abstractmethod
    def get_potential(self):
        pass

    @abstractmethod
    def compute_properties(self) -> Dict:
        """Property order is expected to be equal to implemented_properties"""
        pass

    def calculate(self, atoms=None, **kwargs):
        self.update(atoms)
        properties = self.compute_properties()
        results = self._build_result(properties)

        if not self._verify_results(results):
            raise RuntimeError("Not all implemented properties are returned")
        self.results = results

    def _build_result(self, properties):
        results = {}
        for k, p in zip(self.implemented_properties, properties):
            results[k] = p
            print(k, p)
        return results

    def _verify_results(self, results) -> bool:
        return all([p in results for p in self.implemented_properties])



    # ase plumbing

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
