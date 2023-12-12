# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import ffsim
import numpy as np
from asv_runner.benchmarks.mark import skip_for_params

from qiskit_cold_atom.fermions import FermionSimulator, Hop, Interaction, Phase
from qiskit_cold_atom.fermions.ffsim_backend import FfsimBackend

FERMION_SIMULATOR_CONFIG = {
    "backend_name": "fermion_simulator",
    "backend_version": "0.0.1",
    "n_qubits": 100,
    "basis_gates": None,
    "gates": [],
    "local": False,
    "simulator": True,
    "conditional": False,
    "open_pulse": False,
    "memory": True,
    "max_shots": 1e6,
    "coupling_map": None,
    "description": "a base simulator for fermionic circuits. Instead of qubits, each wire represents"
    " a single fermionic mode",
    "supported_instructions": None,
}


class FfsimBackendBenchmark:
    """Benchmark ffsim backend."""

    param_names = [
        "norb",
        "filling_fraction",
    ]
    params = [
        (4, 8, 12, 16),
        (0.25, 0.5),
    ]

    def setup(self, norb: int, filling_fraction: float):
        self.norb = norb
        nocc = int(norb * filling_fraction)
        self.nelec = (nocc, nocc)

        n_layers = 5
        self.shots = 10_000
        occ_a = [1] * nocc + [0] * (norb - nocc)
        occ_b = [1] * nocc + [0] * (norb - nocc)
        occupations = [occ_a, occ_b]

        rng = np.random.default_rng()
        hopping_coeffs = rng.standard_normal(size=(n_layers, norb - 1))
        interaction_coeffs = rng.standard_normal(size=n_layers)
        phase_coeffs = rng.standard_normal(size=(n_layers, norb))

        self.fermion_simulator_backend = FermionSimulator(FERMION_SIMULATOR_CONFIG)
        self.ffsim_backend = FfsimBackend()

        self.circuit = self.fermion_simulator_backend.initialize_circuit(occupations)
        for hopping, interaction, mu in zip(hopping_coeffs, interaction_coeffs, phase_coeffs):
            self.circuit.append(Hop(2 * norb, hopping), list(range(2 * norb)))
            self.circuit.append(Interaction(2 * norb, interaction), list(range(2 * norb)))
            self.circuit.append(Phase(2 * norb, mu), list(range(2 * norb)))
        self.circuit.measure_all()

        ffsim.init_cache(self.norb, self.nelec)

    @skip_for_params([(8, 0.5), (12, 0.25), (12, 0.5), (16, 0.25), (16, 0.5)])
    def time_simulate_fermion_simulator(self, *_):
        job = self.fermion_simulator_backend.run(self.circuit, shots=self.shots, seed=1234)
        _ = job.result()

    def time_simulate_ffsim(self, *_):
        job = self.ffsim_backend.run(self.circuit, shots=self.shots, seed=1234)
        _ = job.result()
