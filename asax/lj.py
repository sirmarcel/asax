from .calculator import Calculator


class LennardJones(Calculator):
    """Lennard-Jones Potential"""

    implemented_properties = [
        "energy",
        "forces",
        "stress",
        # "stresses",
        # "energies",
    ]

    def __init__(self, epsilon=1.0, sigma=1.0, rc=None, ro=None, **kwargs):
        """
        Paramters:
            sigma: The potential minimum is at  2**(1/6) * sigma, default 1.0
            epsilon: The potential depth, default 1.0
            rc: Cut-off for the NeighborList is set to 3 * sigma if None, the default.
            ro: Onset of the cutoff function. Set to 0.8*rc if None, the default.
            x64: Determine if double precision is used. Default to True.

        """

        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.sigma = sigma

        if rc is None:
            rc = 3 * self.sigma
        self.rc = rc

        if ro is None:
            ro = 0.8 * self.rc
        self.ro = ro

    def get_energy(self, displacement):
        return get_lj(displacement, self.epsilon, self.sigma, self.ro, self.rc)


def get_lj(displacement, epsilon, sigma, ro, rc):
    from jax_md import energy

    return energy.lennard_jones_pair(
        displacement,
        sigma=sigma,
        epsilon=epsilon,
        r_onset=ro / sigma,
        r_cutoff=rc / sigma,
    )
