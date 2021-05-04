from ase import units
from ase.md import VelocityVerlet
from ase.build import bulk
import time
from ase.calculators.lj import LennardJones as aseLJ
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

sigma = 2.0
epsilon = 1.5
rc = 10.0
ro = 6.0

atoms = bulk("Ar", cubic=True) * [5, 5, 5]
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)

atoms.calc = aseLJ(epsilon=epsilon, sigma=sigma, rc=rc, ro=ro, smooth=True)

steps = 1000
start = time.monotonic()

dyn = VelocityVerlet(atoms, timestep=5.0 * units.fs)
dyn.run(steps)

elapsed_seconds = round(time.monotonic() - start, 2)
step_time_ms = round((elapsed_seconds / steps) * 1000, 2)

print("{} steps".format(steps))
print("{} seconds total".format(elapsed_seconds))
print("{} ms/step".format(step_time_ms))


