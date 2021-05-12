from asax.jax_utils import initialize_cubic_argon
from ase import units
from ase.atoms import Atoms
from ase.md import VelocityVerlet
import time
from ase.calculators.lj import LennardJones as aseLJ

class AseNve:
    step_time_ms: float

    def __init__(self, atoms: Atoms, dt: float):
        self.atoms = atoms
        # TODO: Implement dt parameter
        self.dyn = VelocityVerlet(atoms, timestep=dt)

    def run(self, steps: int):
        self.step_time_ms = None
        start = time.monotonic()
        self.dyn.run(steps)
        elapsed = round(time.monotonic() - start, 2)
        self.step_time_ms = round(elapsed/steps * 1000, 2)


steps = 10000
atoms = initialize_cubic_argon()

md = AseNve(atoms, 5.0 * units.fs)
md.run(steps)

print("{} steps: {} ms/step".format(steps, md.step_time_ms))

# 100 steps: 160.7 ms/step
# 500 steps: 73.4 ms/step
# 1000 steps: 62.19 ms/step
# 2000 steps: 62.81 ms/step
# 10.000 steps: 
