import numpy as np

from ase.calculators.abc import GetPropertiesMixin
from ase.calculators.calculator import compare_atoms

from asax import utils


class Calculator(GetPropertiesMixin):
    def __init__(self, x64=True):
        self.x64 = x64
        self.atoms = None
        self.results = {}

        from jax.config import config
        config.update("jax_enable_x64", x64)

    def update(self, atoms):
        if atoms is None and self.atoms is None:
            raise RuntimeError("Need an Atoms object to do anything!")

        if self.atoms is None:
            self.atoms = atoms.copy()
            self.results = {}
            self.setup()

        else:
            changes = compare_atoms(self.atoms, atoms)
            if changes:
                self.results = {}

                if "cell" in changes:
                    self.atoms = None
                    self.update(atoms)
                else:
                    self.atoms = atoms

    def setup(self):
        displacement = utils.get_displacement(self.atoms)
        self.potential = utils.get_potential(self.get_energy(displacement))

    def calculate(self, atoms=None, **kwargs):
        self.update(atoms)

        R = self.atoms.get_positions()

        energy, grad = self.potential(R)

        self.results["energy"] = float(energy)
        self.results["forces"] = np.asarray(-grad)

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
