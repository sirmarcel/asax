from abc import ABC, abstractmethod
import numpy as np
from jax.config import config
from jax_md import space
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

        self.atoms = None
        self.results = {}

    def update(self, atoms):
        if atoms is None and self.atoms is None:
            raise RuntimeError("Need an Atoms object to do anything!")

        if self.atoms is None:
            self.atoms = atoms.copy()
            self.results = {}
            self.setup()
            return

        changes = compare_atoms(self.atoms, atoms)
        if not changes:
            return

        if changes:
            # TODO: verify that "if not changes" works as expected
            print("changes detected!")

        self.results = {}
        if "cell" in changes:
            self.atoms = None
            self.update(atoms)
            return

        self.atoms = atoms

    def setup(self):
        # TODO: jit displacement?
        self.displacement = utils.get_displacement(self.atoms)
        self.potential = self.get_potential()

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
    def compute_properties(self):
        pass

    def calculate(self, atoms=None, **kwargs):
        self.update(atoms)
        self.results = self.compute_properties()

    # ase plumbing

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(f"{name} property not implemented")

        self.update(atoms)

        print(self.results)

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
