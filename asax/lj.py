from typing import Tuple

import jax.numpy as jnp
from ase import units
from jax_md import energy
from jax_md.energy import NeighborFn
from jax_md.partition import NeighborListFormat

from .calculator import Calculator
from .jax_utils import EnergyFn


class LennardJones(Calculator):
    """Lennard-Jones Potential"""

    implemented_properties = ["energy", "energies", "forces", "stress"]

    def __init__(
        self,
        epsilon=1.0,
        sigma=1.0,
        rc=None,
        ro=None,
        stress=False,
        dr_threshold=1 * units.Angstrom,
        **kwargs
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
            sigma=jnp.array(self.sigma, dtype=self.global_dtype),
            epsilon=jnp.array(self.epsilon, dtype=self.global_dtype),
            r_onset=jnp.array(normalized_ro, dtype=self.global_dtype),
            r_cutoff=jnp.array(normalized_rc, dtype=self.global_dtype),
            per_particle=True,
            dr_threshold=self.dr_threshold,
            format=NeighborListFormat.Dense
        )
