from abc import ABC, abstractmethod
import numpy as np
from jax.config import config

from ase.calculators.abc import GetPropertiesMixin
from ase.calculators.calculator import compare_atoms, PropertyNotImplementedError
from ase.constraints import full_3x3_to_voigt_6_stress

from asax import utils


class Calculator(ABC, GetPropertiesMixin):

    # TODO: Can this be abstract?
    implemented_properties = ["energy", "forces"]

    def __init__(self, x64=True):
        self.x64 = x64
        config.update("jax_enable_x64", self.x64)

        self.atoms = None
        self.box: np.array = None
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
        if not changes: return

        if changes:
            # TODO: verify that if not changes works as expected
            print("changes detected!")

        self.results = {}
        if "cell" in changes:
            self.atoms = None
            self.update(atoms)
            return
        
        self.atoms = atoms
        self.box = self.atoms.get_cell().array
        

    def setup(self):
        self.displacement = utils.get_displacement(self.atoms)
        self.potential = self.get_potential()

    @property
    def R(self):
        return self.atoms.get_positions()


    @abstractmethod
    def get_potential(self):
        pass

    
    @abstractmethod
    def compute_properties(self):
        pass
    

    def calculate(self, atoms=None, **kwargs):
        self.update(atoms)

        results = self.compute_properties()
        # TODO: Assert that all implemented properties are contained in the results dictioniary
        self.results = results
        


        # R = self.atoms.get_positions()

        # if not self.stress:
        #     energy, grad = self.potential(R)
        # else:
        #     energy, gradients = self.potential(R)
        #     grad, stress = gradients

        # self.results["energy"] = float(energy)
        # self.results["forces"] = np.asarray(-grad)

        # if self.stress:
        #     self.results["stress"] = full_3x3_to_voigt_6_stress(
        #         np.asarray(stress) / self.atoms.get_volume()
        #     )



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

    
    def get_stress(self, atoms=None):
        # TODO
        pass
