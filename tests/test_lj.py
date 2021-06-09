from unittest import TestCase
from ase.calculators.lj import LennardJones as aLJ
from ase import Atoms
from asax.lj import LennardJones as jLJ
from allclose import AllClose
from ase.calculators.calculator import PropertyNotImplementedError


class TestCalculator(TestCase):
    def test_stress(self):
        from ase.build import bulk

        atoms = bulk("Ar", cubic=True)
        calculator = jLJ(stress=False)
        self.assertRaises(PropertyNotImplementedError, calculator.get_stress, atoms)

        calculator = jLJ(stress=True)
        calculator.get_stress(atoms)


class TestLennardJonesAgainstASE(TestCase, AllClose):
    def setUp(self):
        sigma = 2.0
        epsilon = 1.5
        rc = 10.0
        ro = 6.0

        self.j = jLJ(epsilon=epsilon, sigma=sigma, rc=rc, ro=ro, x64=True)
        self.a = aLJ(epsilon=epsilon, sigma=sigma, rc=rc, ro=ro, smooth=True)
        self.j_stress = jLJ(
            epsilon=epsilon, sigma=sigma, rc=rc, ro=ro, x64=True, stress=True
        )

    def test_twobody(self):
        atoms = Atoms(positions=[[0, 0, 0], [8, 0, 0]])

        self.assertAllClose(
            self.a.get_potential_energy(atoms), self.j.get_potential_energy(atoms)
        )
        self.assertAllClose(self.a.get_forces(atoms), self.j.get_forces(atoms))
        self.assertRaises(PropertyNotImplementedError, self.j.get_stress, atoms)

    def test_threebody(self):
        atoms = Atoms(positions=[[0, 0, 0], [2, 0, 0], [1, 1, 1]])

        self.assertAllClose(
            self.a.get_potential_energy(atoms), self.j.get_potential_energy(atoms)
        )
        self.assertAllClose(self.a.get_forces(atoms), self.j.get_forces(atoms))
        self.assertRaises(PropertyNotImplementedError, self.j.get_stress, atoms)

    def test_solid_cubic(self):
        from ase.build import bulk

        atoms = bulk("Ar", cubic=True) * [5, 5, 5]
        atoms.set_cell(1.05 * atoms.get_cell(), scale_atoms=True)

        self.assertAllClose(
            self.a.get_potential_energy(atoms), self.j.get_potential_energy(atoms)
        )
        self.assertAllClose(
            self.a.get_forces(atoms), self.j.get_forces(atoms), atol=1e-14
        )
        self.assertAllClose(self.a.get_stress(atoms), self.j_stress.get_stress(atoms))

    def test_solid_noncubic(self):
        from ase.build import bulk

        atoms = bulk("Ar", cubic=False) * [9, 9, 9]
        atoms.set_cell(1.05 * atoms.get_cell(), scale_atoms=True)

        self.assertAllClose(
            self.a.get_potential_energy(atoms), self.j.get_potential_energy(atoms)
        )
        self.assertAllClose(
            self.a.get_forces(atoms), self.j.get_forces(atoms), atol=1e-14
        )
        self.assertAllClose(self.a.get_stress(atoms), self.j_stress.get_stress(atoms))
