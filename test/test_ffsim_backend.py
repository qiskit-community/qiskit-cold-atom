# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""ffsim backend tests."""

from __future__ import annotations

import math
import unittest

import numpy as np
from qiskit.test import QiskitTestCase

from qiskit_cold_atom.fermions import (
    FermiHubbard,
    FermionSimulator,
    Hop,
    Interaction,
    Phase,
    FRXGate,
    FRYGate,
    FRZGate,
)

try:
    from qiskit_cold_atom.fermions.ffsim_backend import FfsimBackend

    HAVE_FFSIM = True
except ImportError:
    HAVE_FFSIM = False


def _random_occupations(norb: int, nelec: tuple[int, int], seed=None):
    rng = np.random.default_rng(seed)
    n_alpha, n_beta = nelec
    alpha_bits = np.zeros(norb, dtype=int)
    alpha_bits[:n_alpha] = 1
    beta_bits = np.zeros(norb, dtype=int)
    beta_bits[:n_beta] = 1
    rng.shuffle(alpha_bits)
    rng.shuffle(beta_bits)
    return [list(alpha_bits), list(beta_bits)]


def _fidelity(counts1: dict[str, int], counts2: dict[str, int]) -> float:
    result = 0
    shots = sum(counts1.values())
    assert sum(counts2.values()) == shots
    for bitstring in counts1.keys() | counts2.keys():
        prob1 = counts1.get(bitstring, 0) / shots
        prob2 = counts2.get(bitstring, 0) / shots
        result += math.sqrt(prob1 * prob2)
    return result**2


@unittest.skipUnless(HAVE_FFSIM, "requires ffsim")
class TestFfsimBackend(QiskitTestCase):
    """Test FfsimBackend."""

    def test_hop_gate_spinless(self):
        """Test hop gate, spinless."""
        norb = 6
        nelec = 3

        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        rng = np.random.default_rng()
        for _ in range(3):
            occupations = _random_occupations(norb, (nelec, 0), seed=rng)[0]
            hopping = rng.standard_normal(norb // 2 - 1)

            qc = sim_backend.initialize_circuit(occupations)
            qc.append(Hop(norb, hopping), list(range(norb)))
            job = sim_backend.run(qc)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

            # test acting on subset of orbitals
            qc = sim_backend.initialize_circuit(occupations)
            qubits = rng.choice(np.arange(norb), norb - 2, replace=False)
            qc.append(Hop(norb - 2, hopping[: (norb - 2) // 2 - 1]), list(qubits))
            job = sim_backend.run(qc)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

    def test_hop_gate(self):
        """Test hop gate."""
        norb = 5
        nelec = (3, 2)

        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        rng = np.random.default_rng()
        for _ in range(3):
            occupations = _random_occupations(norb, nelec, seed=rng)
            hopping = rng.standard_normal(norb - 1)

            qc = sim_backend.initialize_circuit(occupations)
            qc.append(Hop(2 * norb, hopping), list(range(2 * norb)))
            job = sim_backend.run(qc, num_species=2)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc, num_species=2)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

            # test acting on subset of orbitals
            qc = sim_backend.initialize_circuit(occupations)
            orbs = rng.choice(np.arange(norb), norb - 1, replace=False)
            qubits = np.concatenate([orbs, orbs + norb])
            qc.append(Hop(2 * (norb - 1), hopping[: norb - 2]), list(qubits))
            job = sim_backend.run(qc, num_species=2)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc, num_species=2)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

    def test_hop_gate_sign(self):
        """Test hop gate correctly handles fermionic sign."""
        norb = 4

        occupations = [[1, 0, 1, 0], [1, 0, 1, 0]]
        hopping = [1.0]

        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        qc = sim_backend.initialize_circuit(occupations)
        orbs = np.array([1, 0])
        qubits = np.concatenate([orbs, orbs + norb])
        qc.append(Hop(4, hopping), list(qubits))
        job = sim_backend.run(qc, num_species=2)
        expected_vec = job.result().get_statevector()
        job = ffsim_backend.run(qc, num_species=2)
        ffsim_vec = job.result().get_statevector()
        np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

    def test_interaction_gate_spinless(self):
        """Test interaction gate, spinless."""
        norb = 6
        nelec = 3

        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        rng = np.random.default_rng()
        for _ in range(3):
            occupations = _random_occupations(norb, (nelec, 0), seed=rng)[0]
            interaction = rng.standard_normal()

            qc = sim_backend.initialize_circuit(occupations)
            qc.append(Interaction(norb, interaction), list(range(norb)))
            job = sim_backend.run(qc)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

            # # test acting on subset of orbitals
            qc = sim_backend.initialize_circuit(occupations)
            qubits = rng.choice(np.arange(norb), norb - 2, replace=False)
            qc.append(Interaction(norb - 2, interaction), list(qubits))
            job = sim_backend.run(qc)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

    def test_interaction_gate(self):
        """Test interaction gate."""
        norb = 5
        nelec = (3, 2)

        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        rng = np.random.default_rng()
        for _ in range(3):
            occupations = _random_occupations(norb, nelec, seed=rng)
            interaction = rng.standard_normal()

            qc = sim_backend.initialize_circuit(occupations)
            qc.append(Interaction(2 * norb, interaction), list(range(2 * norb)))
            job = sim_backend.run(qc, num_species=2)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc, num_species=2)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

            # test acting on subset of orbitals
            qc = sim_backend.initialize_circuit(occupations)
            orbs = rng.choice(np.arange(norb), norb - 1, replace=False)
            qubits = np.concatenate([orbs, orbs + norb])
            qc.append(Interaction(2 * (norb - 1), interaction), list(qubits))
            job = sim_backend.run(qc, num_species=2)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc, num_species=2)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

    def test_phase_gate_spinless(self):
        """Test phase gate."""
        norb = 6
        nelec = 3

        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        rng = np.random.default_rng()
        for _ in range(3):
            occupations = _random_occupations(norb, (nelec, 0), seed=rng)[0]
            mu = rng.standard_normal(norb // 2)

            qc = sim_backend.initialize_circuit(occupations)
            qc.append(Phase(norb, mu), list(range(norb)))
            job = sim_backend.run(qc)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

            # test acting on subset of orbitals
            qc = sim_backend.initialize_circuit(occupations)
            qubits = rng.choice(np.arange(norb), norb - 2, replace=False)
            qc.append(Phase(norb - 2, mu[: (norb - 2) // 2]), list(qubits))
            job = sim_backend.run(qc)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

    def test_phase_gate(self):
        """Test phase gate."""
        norb = 5
        nelec = (3, 2)

        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        rng = np.random.default_rng()
        for _ in range(3):
            occupations = _random_occupations(norb, nelec, seed=rng)
            mu = rng.standard_normal(norb)

            qc = sim_backend.initialize_circuit(occupations)
            qc.append(Phase(2 * norb, mu), list(range(2 * norb)))
            job = sim_backend.run(qc, num_species=2)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc, num_species=2)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

            # test acting on subset of orbitals
            qc = sim_backend.initialize_circuit(occupations)
            orbs = rng.choice(np.arange(norb), norb - 1, replace=False)
            qubits = np.concatenate([orbs, orbs + norb])
            qc.append(Phase(2 * (norb - 1), mu[: norb - 1]), list(qubits))
            job = sim_backend.run(qc, num_species=2)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc, num_species=2)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

    def test_frz_gate(self):
        """Test FRZGate."""
        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        occupations = [1, 0, 0, 1, 0, 1, 0, 1]
        qc = sim_backend.initialize_circuit(occupations)
        qc.append(FRZGate(0.25), [0, 4])
        qc.append(FRZGate(0.5), [1, 5])
        job = sim_backend.run(qc)
        expected_vec = job.result().get_statevector()
        job = ffsim_backend.run(qc)
        ffsim_vec = job.result().get_statevector()
        np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

        occupations = [[1, 0, 0, 1], [0, 1, 0, 1]]
        qc = sim_backend.initialize_circuit(occupations)
        qc.append(FRZGate(0.25), [0, 4])
        qc.append(FRZGate(0.5), [1, 5])
        job = sim_backend.run(qc, num_species=2)
        expected_vec = job.result().get_statevector()
        job = ffsim_backend.run(qc, num_species=2)
        ffsim_vec = job.result().get_statevector()
        np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

    def test_frx_gate(self):
        """Test FRXGate."""
        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        occupations = [1, 0, 0, 1, 0, 1, 0, 1]
        qc = sim_backend.initialize_circuit(occupations)
        qc.append(FRXGate(0.25), [0, 4])
        qc.append(FRXGate(0.5), [1, 5])
        job = sim_backend.run(qc)
        expected_vec = job.result().get_statevector()
        job = ffsim_backend.run(qc)
        ffsim_vec = job.result().get_statevector()
        np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

        occupations = [[1, 0, 0, 1], [0, 1, 0, 1]]
        qc = sim_backend.initialize_circuit(occupations)
        qc.append(FRXGate(0.25), [0, 4])
        qc.append(FRXGate(0.5), [1, 5])
        with self.assertRaisesRegex(RuntimeError, "num_species"):
            job = ffsim_backend.run(qc, num_species=2)
            ffsim_vec = job.result().get_statevector()

    def test_fry_gate(self):
        """Test FRYGate."""
        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        occupations = [1, 0, 0, 1, 0, 1, 0, 1]
        qc = sim_backend.initialize_circuit(occupations)
        qc.append(FRYGate(0.25), [0, 4])
        qc.append(FRYGate(0.5), [1, 5])
        job = sim_backend.run(qc)
        expected_vec = job.result().get_statevector()
        job = ffsim_backend.run(qc)
        ffsim_vec = job.result().get_statevector()
        np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

        occupations = [[1, 0, 0, 1], [0, 1, 0, 1]]
        qc = sim_backend.initialize_circuit(occupations)
        qc.append(FRYGate(0.25), [0, 4])
        qc.append(FRYGate(0.5), [1, 5])
        with self.assertRaisesRegex(RuntimeError, "num_species"):
            job = ffsim_backend.run(qc, num_species=2)
            ffsim_vec = job.result().get_statevector()

    def test_fermi_hubbard_gate_spinless(self):
        """Test Fermi-Hubbard gate."""
        norb = 6
        nelec = 3

        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        rng = np.random.default_rng()
        for _ in range(3):
            occupations = _random_occupations(norb, (nelec, 0), seed=rng)[0]
            hopping = rng.standard_normal(norb // 2 - 1)
            interaction = rng.standard_normal()
            mu = rng.standard_normal(norb // 2)

            qc = sim_backend.initialize_circuit(occupations)
            qc.append(FermiHubbard(norb, hopping, interaction, mu), list(range(norb)))
            job = sim_backend.run(qc)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

            # test acting on subset of orbitals
            qc = sim_backend.initialize_circuit(occupations)
            qubits = rng.choice(np.arange(norb), norb - 2, replace=False)
            qc.append(
                FermiHubbard(
                    norb - 2,
                    hopping[: (norb - 2) // 2 - 1],
                    interaction,
                    mu[: (norb - 2) // 2],
                ),
                list(qubits),
            )
            job = sim_backend.run(qc)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

    def test_fermi_hubbard_gate(self):
        """Test Fermi-Hubbard gate."""
        norb = 5
        nelec = (3, 2)

        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        rng = np.random.default_rng()
        for _ in range(3):
            occupations = _random_occupations(norb, nelec, seed=rng)
            hopping = rng.standard_normal(norb - 1)
            interaction = rng.standard_normal()
            mu = rng.standard_normal(norb)

            qc = sim_backend.initialize_circuit(occupations)
            qc.append(
                FermiHubbard(2 * norb, hopping, interaction, mu), list(range(2 * norb))
            )
            job = sim_backend.run(qc, num_species=2)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc, num_species=2)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

            # test acting on subset of orbitals
            qc = sim_backend.initialize_circuit(occupations)
            orbs = rng.choice(np.arange(norb), norb - 1, replace=False)
            qubits = np.concatenate([orbs, orbs + norb])
            qc.append(
                FermiHubbard(
                    2 * (norb - 1), hopping[: norb - 2], interaction, mu[: norb - 1]
                ),
                list(qubits),
            )
            job = sim_backend.run(qc, num_species=2)
            expected_vec = job.result().get_statevector()
            job = ffsim_backend.run(qc, num_species=2)
            ffsim_vec = job.result().get_statevector()
            np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

    def test_fermi_hubbard_gate_simple(self):
        """Test a simple Fermi-Hubbard gate."""
        norb = 4

        occupations = [[1, 1, 0, 0], [1, 1, 0, 0]]
        hopping = np.arange(norb - 1)
        interaction = 1.0
        mu = np.arange(norb)

        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        qc = sim_backend.initialize_circuit(occupations)
        qc.append(
            FermiHubbard(2 * norb, hopping, interaction, mu), list(range(2 * norb))
        )
        job = sim_backend.run(qc, num_species=2)
        expected_vec = job.result().get_statevector()
        job = ffsim_backend.run(qc, num_species=2)
        ffsim_vec = job.result().get_statevector()
        np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)

    def test_simulate(self):
        """Test simulating and measuring a statevector."""
        norb = 5
        nelec = (3, 2)

        rng = np.random.default_rng(1234)
        occupations = _random_occupations(norb, nelec, seed=rng)
        hopping = rng.standard_normal(norb - 1)
        interaction = rng.standard_normal()
        mu = rng.standard_normal(norb)

        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        qc = sim_backend.initialize_circuit(occupations)
        qc.append(Hop(2 * norb, hopping), list(range(2 * norb)))
        qc.append(Interaction(2 * norb, interaction), list(range(2 * norb)))
        qc.append(Phase(2 * norb, mu), list(range(2 * norb)))
        qc.measure_all()

        job = sim_backend.run(qc, shots=10000, seed=1234, num_species=2)
        result = job.result()
        expected_vec = result.get_statevector()
        expected_counts = result.get_counts()

        job = ffsim_backend.run(qc, shots=10000, seed=1234, num_species=2)
        result = job.result()
        ffsim_vec = result.get_statevector()
        ffsim_counts = result.get_counts()

        np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)
        assert _fidelity(ffsim_counts, expected_counts) > 0.99

    def test_simulate_spinless(self):
        """Test simulating and measuring a statevector."""
        norb = 6
        nelec = 3

        sim_backend = FermionSimulator()
        ffsim_backend = FfsimBackend()

        rng = np.random.default_rng(1234)
        occupations = _random_occupations(norb, (nelec, 0), seed=rng)[0]
        hopping = rng.standard_normal(norb // 2 - 1)
        interaction = rng.standard_normal()
        mu = rng.standard_normal(norb // 2)

        qc = sim_backend.initialize_circuit(occupations)
        qc.append(Hop(norb, hopping), list(range(norb)))
        qc.append(Interaction(norb, interaction), list(range(norb)))
        qc.append(Phase(norb, mu), list(range(norb)))
        qc.measure_all()

        job = sim_backend.run(qc, shots=10000, seed=1234)
        result = job.result()
        expected_vec = result.get_statevector()
        expected_counts = result.get_counts()

        job = ffsim_backend.run(qc, shots=10000, seed=1234)
        result = job.result()
        ffsim_vec = result.get_statevector()
        ffsim_counts = result.get_counts()

        np.testing.assert_allclose(ffsim_vec, expected_vec, atol=1e-12)
        assert _fidelity(ffsim_counts, expected_counts) > 0.99
