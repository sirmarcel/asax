from typing import Dict, List, Tuple

from jax_md.energy import NeighborFn

from .calculator import Calculator
import numpy as np
from asax import jax_utils
from jax import jit
import jax.numpy as jnp
from jax_md import energy, partition
from ase.constraints import full_3x3_to_voigt_6_stress
from ase import units

from .jax_utils import EnergyFn


class LennardJones(Calculator):
    """Lennard-Jones Potential"""

    implemented_properties = ["energy", "energies", "forces", "stress"]

    def __init__(
        self, epsilon=1.0, sigma=1.0, rc=None, ro=None, stress=False, dr_threshold=1 * units.Angstrom, **kwargs
    ):
        """
        Parameters:
            sigma: The potential minimum is at  2**(1/6) * sigma, default 1.0
            epsilon: The potential depth, default 1.0
            rc: Cut-off for the NeighborList is set to 3 * sigma if None, the default.
            ro: Onset of the cutoff function. Set to 0.8*rc if None, the default.
            stress: Compute stress tensor (periodic systems only)
        """
        super().__init__(**kwargs, stress=stress)
        self.epsilon = epsilon
        self.sigma = sigma

        if rc is None:
            rc = 3 * self.sigma
        self.rc = rc

        if ro is None:
            ro = 0.8 * self.rc
        self.ro = ro
        self.dr_threshold = dr_threshold

    def get_energy_function(self) -> Tuple[NeighborFn, EnergyFn]:
        normalized_ro = self.ro / self.sigma
        normalized_rc = self.rc / self.sigma

        return energy.lennard_jones_neighbor_list(
            self.displacement,
            self.box,
            sigma=jnp.float32(self.sigma),
            epsilon=jnp.float32(self.epsilon),
            r_onset=jnp.float32(normalized_ro),
            r_cutoff=jnp.float32(normalized_rc),
            per_particle=True,
            dr_threshold=self.dr_threshold
        )
