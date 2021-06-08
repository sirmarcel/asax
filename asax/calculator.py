from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
from jax.config import config
from jax_md import space
from ase.atoms import Atoms
from ase.calculators.abc import GetPropertiesMixin
from ase.calculators.calculator import compare_atoms, PropertyNotImplementedError
from asax import utils, jax_utils


class Calculator(GetPropertiesMixin, ABC):
    implemented_properties = ["energy", "forces"]

    displacement: space.DisplacementFn
    shift: space.ShiftFn
    potential: jax_utils.PotentialFn

    def __init__(self, x64=True):
        self.x64 = x64
        config.update("jax_enable_x64", self.x64)

        self.atoms: Atoms = None
        self.results = {}

    def update(self, atoms: Atoms):
        if atoms is None and self.atoms is None:
            raise RuntimeError("Need an Atoms object to do anything!")

        if self.atoms is None:
            # new or reset calculator: redo everything
            self.on_atoms_changed(atoms)
            self.atoms = atoms.copy()  # avoid missing changes
            self.results = {}
            self.setup()
            return

        changes = compare_atoms(self.atoms, atoms)
        if changes:
            # any change invalidates results
            self.results = {}

            if "cell" in changes or "pbc" in changes:
                # need to recreate displacement; might as well start over
                self.atoms = None
                self.update(atoms)

            # no major changes, let subclass deal with it
            self.on_atoms_changed(atoms)
            self.atoms = atoms.copy()

    @abstractmethod
    def on_atoms_changed(self, atoms: Atoms):
        """Called whenever a new atoms object is passed so that child classes can react"""
        pass

    def setup(self):
        """Create displacement, shift and potential"""
        self.displacement, self.shift = self.get_displacement(self.atoms)
        self.potential = self.get_potential()

    def get_displacement(self, atoms: Atoms):
        if not all(atoms.get_pbc()):
            return space.free()

        box = atoms.get_cell().array
        return space.periodic_general(box, fractional_coordinates=False)

    @property
    def R(self):
        return self.atoms.get_positions()

    @property
    def box(self):
        return self.atoms.get_cell().array

    @abstractmethod
    def get_potential(self):
        pass

    @abstractmethod
    def compute_properties(self) -> Dict:
        """Expected to return a dictionary keyed on (a subset of) implemented_properties"""
        pass

    def calculate(self, atoms=None, **kwargs):
        self.update(atoms)
        self.results = self.compute_properties()

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
