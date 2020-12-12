from unittest import TestCase
from ase.calculators.lj import LennardJones as aLJ
from ase import Atoms

from asax.lj import LennardJones as jLJ

from allclose import AllClose


class TestLennardJonesAgainstASE(TestCase, AllClose):
    def setUp(self):
        sigma = 2.0
        epsilon = 1.5
        rc = 11.0
        ro = 10.9

        self.j = jLJ(epsilon=epsilon, sigma=sigma, rc=rc, ro=ro, x64=True)
        self.a = aLJ(epsilon=epsilon, sigma=sigma, rc=rc)

    def test_twobody(self):
        atoms = Atoms(positions=[[0, 0, 0], [2, 0, 0]])

        self.assertAllClose(
            self.a.get_potential_energy(atoms), self.j.get_potential_energy(atoms)
        )
        self.assertAllClose(self.a.get_forces(atoms), self.j.get_forces(atoms))

    def test_threebody(self):
        atoms = Atoms(positions=[[0, 0, 0], [2, 0, 0], [1, 1, 1]])

        self.assertAllClose(
            self.a.get_potential_energy(atoms), self.j.get_potential_energy(atoms)
        )
        self.assertAllClose(self.a.get_forces(atoms), self.j.get_forces(atoms))

    def test_solid_cubic(self):
        from ase.build import bulk

        atoms = bulk("Ar", cubic=True) * [5, 5, 5]
        atoms.set_cell(1.05 * atoms.get_cell(), scale_atoms=True)

        print(atoms.get_cell())

        self.assertAllClose(
            self.a.get_potential_energy(atoms), self.j.get_potential_energy(atoms)
        )
        self.assertAllClose(
            self.a.get_forces(atoms), self.j.get_forces(atoms), atol=1e-14
        )

    def test_solid_noncubic(self):
        from ase.build import bulk

        atoms = bulk("Ar", cubic=False) * [9, 9, 9]
        atoms.set_cell(1.05 * atoms.get_cell(), scale_atoms=True)

        print(atoms.get_cell())

        self.assertAllClose(
            self.a.get_potential_energy(atoms), self.j.get_potential_energy(atoms)
        )
        self.assertAllClose(
            self.a.get_forces(atoms), self.j.get_forces(atoms), atol=1e-14
        )


class TestLennardJonesAgainstASEWithStress(TestCase, AllClose):
    def setUp(self):
        sigma = 2.0
        epsilon = 1.5
        rc = 11.0
        ro = 10.9

        self.j = jLJ(epsilon=epsilon, sigma=sigma, rc=rc, ro=ro, x64=True, stress=True)
        self.a = aLJ(epsilon=epsilon, sigma=sigma, rc=rc)

    def test_solid_cubic(self):
        from ase.build import bulk

        atoms = bulk("Ar", cubic=True) * [5, 5, 5]
        atoms.set_cell(1.05 * atoms.get_cell(), scale_atoms=True)

        print(atoms.get_cell())

        self.assertAllClose(
            self.a.get_potential_energy(atoms), self.j.get_potential_energy(atoms)
        )
        self.assertAllClose(
            self.a.get_forces(atoms), self.j.get_forces(atoms), atol=1e-14
        )
        self.assertAllClose(
            self.a.get_stress(atoms), self.j.get_stress(atoms)
        )

    def test_solid_noncubic(self):
        from ase.build import bulk

        atoms = bulk("Ar", cubic=False) * [9, 9, 9]
        atoms.set_cell(1.05 * atoms.get_cell(), scale_atoms=True)

        print(atoms.get_cell())

        self.assertAllClose(
            self.a.get_potential_energy(atoms), self.j.get_potential_energy(atoms)
        )
        self.assertAllClose(
            self.a.get_forces(atoms), self.j.get_forces(atoms), atol=1e-14
        )
