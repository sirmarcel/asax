from unittest import TestCase
from ase.calculators.lj import LennardJones as aLJ
from ase import Atoms, units
from ase.md import VelocityVerlet

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


class TestLennardJonesAgainstASEinDoublePrecision(TestCase, AllClose):
    def setUp(self):
        sigma = 2.0
        epsilon = 1.5
        rc = 10.0
        ro = 6.0

        # by setting global_dtype to float64, we calculator.py obtains atom positions and the box in the respective data type.
        # as a result, all properties are computed end-to-end in double precision

        self.j = jLJ(epsilon=epsilon, sigma=sigma, rc=rc, ro=ro, x64=True)
        self.j.global_dtype = "float64"

        self.a = aLJ(epsilon=epsilon, sigma=sigma, rc=rc, ro=ro, smooth=True)
        self.j_stress = jLJ(
            epsilon=epsilon, sigma=sigma, rc=rc, ro=ro, x64=True, stress=True
        )
        self.j_stress.global_dtype = "float64"

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
            self.a.get_potential_energy(atoms),
            self.j.get_potential_energy(atoms),
            atol=1e-10,
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
            self.a.get_forces(atoms), self.j.get_forces(atoms), atol=1e-13
        )
        self.assertAllClose(self.a.get_stress(atoms), self.j_stress.get_stress(atoms))


class TestLennardJonesAgainstASEinSinglePrecision(TestCase, AllClose):
    def setUp(self):
        sigma = 2.0
        epsilon = 1.5
        rc = 10.0
        ro = 6.0

        # in the default case, calculator.py obtains atom coordinates and the box in float32.
        # that way, calculations are performed with significant speedup in single precision while JAX is still able to use float64 for reductions.
        # absolute tolerance for assertions are reduced in this case when compared to double precision.

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
            self.a.get_potential_energy(atoms),
            self.j.get_potential_energy(atoms),
            atol=1e-4,
        )
        self.assertAllClose(
            self.a.get_forces(atoms), self.j.get_forces(atoms), atol=1e-4
        )
        self.assertRaises(PropertyNotImplementedError, self.j.get_stress, atoms)

    def test_solid_cubic(self):
        from ase.build import bulk

        atoms = bulk("Ar", cubic=True) * [5, 5, 5]
        atoms.set_cell(1.05 * atoms.get_cell(), scale_atoms=True)

        self.assertAllClose(
            self.a.get_potential_energy(atoms),
            self.j.get_potential_energy(atoms),
            atol=1e-3,
        )
        self.assertAllClose(
            self.a.get_forces(atoms), self.j.get_forces(atoms), atol=1e-6
        )
        self.assertAllClose(
            self.a.get_stress(atoms), self.j_stress.get_stress(atoms), atol=1e-7
        )

    def test_solid_noncubic(self):
        from ase.build import bulk

        atoms = bulk("Ar", cubic=False) * [9, 9, 9]
        atoms.set_cell(1.05 * atoms.get_cell(), scale_atoms=True)

        self.assertAllClose(
            self.a.get_potential_energy(atoms),
            self.j.get_potential_energy(atoms),
            atol=1e-3,
        )
        self.assertAllClose(
            self.a.get_forces(atoms), self.j.get_forces(atoms), atol=5 * 1e-6
        )
        self.assertAllClose(
            self.a.get_stress(atoms), self.j_stress.get_stress(atoms), atol=1e-7
        )


class TestLennardJonesMd(TestCase, AllClose):

    dt = 5 * units.fs
    dr_threshold = 1 * units.Angstrom

    # sigma = 2.0
    # epsilon = 1.5
    # rc = 10.0
    # ro = 6.0

    sigma = 3.40
    epsilon = 0.01042
    rc = 10.54
    ro = 6.0

    def test_nve_against_ase(self):
        from ase.build import bulk

        ase_atoms = bulk("Ar", cubic=True) * [8, 8, 8]
        ase_atoms.calc = aLJ(
            epsilon=self.epsilon, sigma=self.sigma, rc=self.rc, ro=self.ro, smooth=True
        )
        ase_dyn = VelocityVerlet(ase_atoms, timestep=self.dt)

        asax_atoms = bulk("Ar", cubic=True) * [8, 8, 8]
        asax_atoms.calc = jLJ(
            epsilon=self.epsilon, sigma=self.sigma, rc=self.rc, ro=self.ro, x64=True
        )
        asax_dyn = VelocityVerlet(asax_atoms, timestep=self.dt)

        for i in range(10):
            ase_dyn.run(steps=1)
            asax_dyn.run(steps=1)

            self.assertAllClose(
                ase_atoms.get_positions(wrap=True),
                asax_atoms.get_positions(wrap=True),
                atol=1e-8,
                equal_nan=False,
            )
