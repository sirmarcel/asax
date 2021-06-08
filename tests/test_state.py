from unittest import TestCase
from unittest.mock import MagicMock

from ase.build import bulk

# TODO: maybe test abstract class directly?
from asax.lj import LennardJones


class TestState(TestCase):
    def setUp(self):

        self.atoms1 = bulk("Ar", cubic=True)
        self.atoms2 = bulk("Ar", cubic=True)
        self.atoms2.positions[0] += 0.5

        self.atoms3 = bulk("Ar", cubic=True) * [2, 2, 2]

        self.atoms4 = self.atoms3.copy()
        self.atoms4.set_pbc(False)

    def test_on_atoms_changed(self):
        calculator = LennardJones()
        calculator.on_atoms_changed = MagicMock()

        calculator.update(self.atoms1)
        calculator.on_atoms_changed.assert_called_once()

        calculator.on_atoms_changed.reset_mock()
        calculator.update(self.atoms1)
        calculator.on_atoms_changed.assert_not_called()
        calculator.update(self.atoms2)
        calculator.on_atoms_changed.assert_called_once()

    def test_setup(self):
        calculator = LennardJones()
        calculator.setup = MagicMock()

        # same cell + pbc: setup once
        calculator.update(self.atoms1)
        calculator.update(self.atoms1)
        calculator.update(self.atoms2)
        calculator.setup.assert_called_once()

        # different cell: setup again
        calculator.setup.reset_mock()
        calculator.update(self.atoms2)
        calculator.update(self.atoms3)
        calculator.update(self.atoms3)
        calculator.setup.assert_called_once()

        # different pbc: setup again
        calculator.setup.reset_mock()
        calculator.update(self.atoms4)
        calculator.setup.assert_called_once()
