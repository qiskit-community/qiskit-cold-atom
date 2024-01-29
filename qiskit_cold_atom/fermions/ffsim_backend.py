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

"""Fermionic simulator backend that uses ffsim."""

from __future__ import annotations

import datetime
import time
import uuid
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import ffsim  # pylint: disable=import-error
import numpy as np
import scipy.linalg
from qiskit import QuantumCircuit
from qiskit.circuit.library import Barrier, Measure
from qiskit.providers import Options
from qiskit.providers.models import BackendConfiguration
from qiskit.result import Result
from qiskit_aer import AerJob
from qiskit_nature.second_q.operators import FermionicOp
from scipy.sparse.linalg import expm_multiply

from qiskit_cold_atom.circuit_tools import CircuitTools
from qiskit_cold_atom.fermions.base_fermion_backend import BaseFermionBackend
from qiskit_cold_atom.fermions.fermion_gate_library import (
    FermionicGate,
    FRXGate,
    FRYGate,
    FRZGate,
    Hop,
    Interaction,
    LoadFermions,
    Phase,
)


class FfsimBackend(BaseFermionBackend):
    """Fermionic simulator backend that uses ffsim.

    This is a high-performance simulator backend for fermionic circuits that uses `ffsim`_.
    It computes the state vector and simulate measurements with vastly improved efficiency
    compared with the :class:`~.FermionSimulator` backend. Unlike :class:`~.FermionSimulator`,
    it does not compute the full unitary of a circuit.

    Currently, this simulator only supports simulations with 1 or 2 species of fermions.
    The number of fermions of each species is assumed to be preserved, so that the
    dimension of the state vector can be determined from the number of species and the
    number of particles of each species. In particular, when simulating 2 species of fermions,
    gates that mix particles of different species, such as :class:`~.FRXGate` and
    :class:`FRYGate`, are not supported. In this respect, the behavior of this simulator
    differs from :class:`FermionSimulator`, which would automatically resort to a
    single-species simulation in which particles of each species are not distinguished.

    This backend is not supported on Windows, and in order for it to be available,
    Qiskit Cold Atom must be installed with the ``ffsim`` extra, e.g.

    .. code::

        pip install "qiskit-cold-atom[ffsim]"

    .. _ffsim: https://github.com/qiskit-community/ffsim
    """

    _DEFAULT_CONFIGURATION = {
        "backend_name": "ffsim_simulator",
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
        "description": "ffsim simulator for fermionic circuits. Instead of qubits, each wire represents"
        " a single fermionic mode",
        "supported_instructions": None,
    }

    def __init__(self, config_dict: Dict[str, Any] = None, provider=None):
        """Initializing the backend from a configuration dictionary"""

        if config_dict is None:
            config_dict = self._DEFAULT_CONFIGURATION

        super().__init__(
            configuration=BackendConfiguration.from_dict(config_dict), provider=provider
        )

    @classmethod
    def _default_options(cls):
        return Options(shots=1)

    def _execute(self, data: Dict[str, Any], job_id: str = ""):
        """Helper function to execute a job. The circuit and all relevant parameters are given in the
        data dict. Performs validation checks on the received circuits and utilizes
        ffsim to perform the numerical simulations.

         Args:
            data: Data dictionary that that contains the experiments to simulate, given in the shape:
                data = {
                    "num_species": int,
                    "shots": int,
                    "seed": int,
                    "experiments": Dict[str, QuantumCircuit],
                }
            job_id: The job id assigned by the run method

        Returns:
            result: A qiskit job result.
        """
        # Start timer
        start = time.time()

        output = {"results": []}

        num_species = data["num_species"]
        shots = data["shots"]
        seed = data["seed"]

        for exp_i, exp_name in enumerate(data["experiments"]):
            experiment = data["experiments"][exp_name]
            circuit = experiment["circuit"]

            # perform compatibility checks with the backend configuration in case gates and supported
            # instructions are constrained by the backend's configuration
            if self.configuration().gates and self.configuration().supported_instructions:
                CircuitTools.validate_circuits(circuits=circuit, backend=self, shots=shots)

            # check whether all wires are measured
            measured_wires = []

            for inst in circuit.data:
                name = inst[0].name

                if name == "measure":
                    for wire in inst[1]:
                        index = circuit.qubits.index(wire)
                        if index in measured_wires:
                            warnings.warn(
                                f"Wire {index} has already been measured, "
                                f"second measurement is ignored"
                            )
                        else:
                            measured_wires.append(index)

            if measured_wires and len(measured_wires) != len(circuit.qubits):
                warnings.warn(
                    f"Number of wires in circuit ({len(circuit.qubits)}) exceeds number of wires "
                    + f" with assigned measurement instructions ({len(measured_wires)}). "
                    + "This simulator backend only supports measurement of the entire quantum register "
                    "which will instead be performed."
                )

            # If there are no measurements, set shots to None
            if not measured_wires:
                shots = None

            simulation_result = _simulate_ffsim(circuit, num_species, shots, seed)

            output["results"].append(
                {
                    "header": {"name": exp_name, "random_seed": seed},
                    "shots": shots,
                    "status": "DONE",
                    "success": True,
                }
            )
            # add the simulation result at the correct place in the result dictionary
            output["results"][exp_i]["data"] = simulation_result

        output["job_id"] = job_id
        output["date"] = datetime.datetime.now().isoformat()
        output["backend_name"] = self.name()
        output["backend_version"] = self.configuration().backend_version
        output["time_taken"] = time.time() - start
        output["success"] = True
        output["qobj_id"] = None

        return Result.from_dict(output)

    # pylint: disable=arguments-differ, unused-argument
    def run(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        shots: int = 1000,
        seed: Optional[int] = None,
        num_species: int = 1,
        **run_kwargs,
    ) -> AerJob:
        """
        Method to run circuits on the backend.

        Args:
            circuits: QuantumCircuit applying fermionic gates to run on the backend
            shots: Number of measurement shots taken in case the circuit has measure instructions
            seed: seed for the random number generator of the measurement simulation
            num_species: number of different fermionic species described by the circuits
            run_kwargs: Additional keyword arguments that might be passed down when calling
                qiskit.execute() which will have no effect on this backend.

        Returns:
            aer_job: a job object containing the result of the simulation

        Raises:
            ValueError: FfsimBackend only supports num_species=1 or 2.
        """
        if num_species not in (1, 2):
            raise ValueError(f"FfsimBackend only supports num_species=1 or 2. Got {num_species}.")

        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        data = {
            "num_species": num_species,
            "shots": shots,
            "seed": seed,
            "experiments": {},
        }

        for idx, circuit in enumerate(circuits):
            data["experiments"][f"experiment_{idx}"] = {
                "circuit": circuit,
            }

        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._execute, data)
        aer_job.submit()
        return aer_job


def _simulate_ffsim(
    circuit: QuantumCircuit, num_species: int, shots: int | None = None, seed=None
) -> dict[str, Any]:
    assert circuit.num_qubits % num_species == 0
    norb = circuit.num_qubits // num_species
    occ_a, occ_b = _get_initial_occupations(circuit, num_species)
    nelec = len(occ_a), len(occ_b)
    vec = ffsim.slater_determinant(norb, (occ_a, occ_b))
    qubit_indices = {q: i for i, q in enumerate(circuit.qubits)}
    for instruction in circuit.data:
        op, qubits, _ = instruction.operation, instruction.qubits, instruction.clbits
        if isinstance(op, Hop):
            orbs = [qubit_indices[q] for q in qubits]
            spatial_orbs = _get_spatial_orbitals(orbs, norb, num_species)
            vec = _simulate_hop(
                vec,
                np.array(op.params),
                spatial_orbs,
                norb=norb,
                nelec=nelec,
                num_species=num_species,
                copy=False,
            )
        elif isinstance(op, Interaction):
            orbs = [qubit_indices[q] for q in qubits]
            spatial_orbs = _get_spatial_orbitals(orbs, norb, num_species)
            (interaction,) = op.params
            vec = _simulate_interaction(
                vec,
                interaction,
                spatial_orbs,
                norb=norb,
                nelec=nelec,
                num_species=num_species,
                copy=False,
            )
        elif isinstance(op, Phase):
            orbs = [qubit_indices[q] for q in qubits]
            spatial_orbs = _get_spatial_orbitals(orbs, norb, num_species)
            vec = _simulate_phase(
                vec,
                np.array(op.params),
                spatial_orbs,
                norb=norb,
                nelec=nelec,
                num_species=num_species,
                copy=False,
            )
        elif isinstance(op, FRZGate):
            orbs = [qubit_indices[q] for q in qubits]
            # pass num_species=1 here due to the definition of FRZGate
            spatial_orbs = _get_spatial_orbitals(orbs, norb, num_species=1)
            (phi,) = op.params
            vec = _simulate_frz(
                vec,
                phi,
                spatial_orbs,
                norb=norb,
                nelec=nelec,
                num_species=num_species,
                copy=False,
            )
        elif isinstance(op, FRXGate):
            if num_species != 1:
                raise RuntimeError(
                    f"Encountered FRXGate even though num_species={num_species}. "
                    "FRXGate is only supported for num_species=1."
                )
            orbs = [qubit_indices[q] for q in qubits]
            spatial_orbs = _get_spatial_orbitals(orbs, norb, num_species=1)
            (phi,) = op.params
            vec = ffsim.apply_tunneling_interaction(
                vec, -phi, spatial_orbs, norb, nelec, copy=False
            )
        elif isinstance(op, FRYGate):
            if num_species != 1:
                raise RuntimeError(
                    f"Encountered FRXGate even though num_species={num_species}. "
                    "FRXGate is only supported for num_species=1."
                )
            orbs = [qubit_indices[q] for q in qubits]
            spatial_orbs = _get_spatial_orbitals(orbs, norb, num_species=1)
            (phi,) = op.params
            vec = ffsim.apply_givens_rotation(vec, -phi, spatial_orbs, norb, nelec, copy=False)
        elif isinstance(op, FermionicGate):
            orbs = [qubit_indices[q] for q in qubits]
            spatial_orbs = _get_spatial_orbitals(orbs, norb, num_species)
            ferm_op = _fermionic_op_to_fermion_operator(op.generator, spatial_orbs)
            linop = ffsim.linear_operator(ferm_op, norb, nelec)
            # TODO use ferm_op.values once it's available
            scale = sum(abs(ferm_op[k]) for k in ferm_op)
            vec = expm_multiply(-1j * linop, vec, traceA=scale)
        elif isinstance(op, (LoadFermions, Measure, Barrier)):
            # these gates are handled separately or are no-ops
            pass
        else:
            warnings.warn(f"Unrecognized gate type {type(op)}, skipping it...")

    result = {"statevector": vec}

    if shots is None:
        result["memory"] = []
        result["counts"] = {}
    else:
        rng = np.random.default_rng(seed)
        probs = np.abs(vec) ** 2
        samples = rng.choice(np.arange(len(vec)), size=shots, replace=True, p=probs)
        bitstrings = ffsim.indices_to_strings(samples, norb, nelec)
        # flip beta-alpha to alpha-beta ordering
        bitstrings = [f"{b[len(b) // 2 :]}{b[: len(b) // 2]}" for b in bitstrings]
        # remove bits from absent spins
        bitstrings = [b[: num_species * norb] for b in bitstrings]
        result["memory"] = bitstrings
        result["counts"] = Counter(bitstrings)

    return result


def _get_initial_occupations(circuit: QuantumCircuit, num_species: int):
    norb = circuit.num_qubits // num_species
    occ_a, occ_b = set(), set()
    occupations = [occ_a, occ_b]
    active_qubits = set()
    for instruction in circuit.data:
        if isinstance(instruction.operation, LoadFermions):
            for q in instruction.qubits:
                if q in active_qubits:
                    raise ValueError(
                        f"Encountered Load instruction on qubit {q} after it has "
                        "already been operated on."
                    )
                spin, orb = divmod(circuit.qubits.index(q), norb)
                # reverse index due to qiskit convention
                occupations[spin].add(norb - 1 - orb)
        else:
            active_qubits |= set(instruction.qubits)
    return tuple(occ_a), tuple(occ_b)


def _get_spatial_orbitals(orbs: list[int], norb: int, num_species: int) -> list[int]:
    assert len(orbs) % num_species == 0
    alpha_orbs = orbs[: len(orbs) // num_species]
    if num_species == 2:
        beta_orbs = [orb - norb for orb in orbs[len(orbs) // 2 :]]
        assert alpha_orbs == beta_orbs
    # reverse orbitals due to qiskit convention
    alpha_orbs = [norb - 1 - orb for orb in alpha_orbs]
    return alpha_orbs


def _simulate_hop(
    vec: np.ndarray,
    coeffs: np.ndarray,
    target_orbs: list[int],
    norb: int,
    nelec: tuple[int, int],
    num_species: int,
    copy: bool,
) -> np.ndarray:
    if num_species == 1:
        return _simulate_hop_spinless(
            vec=vec,
            coeffs=coeffs,
            target_orbs=target_orbs,
            norb=norb,
            nelec=nelec,
            copy=copy,
        )
    else:  # num_species == 2
        return _simulate_hop_spinful(
            vec=vec,
            coeffs=coeffs,
            target_orbs=target_orbs,
            norb=norb,
            nelec=nelec,
            copy=copy,
        )


def _simulate_hop_spinless(
    vec: np.ndarray,
    coeffs: np.ndarray,
    target_orbs: list[int],
    norb: int,
    nelec: tuple[int, int],
    copy: bool,
) -> np.ndarray:
    assert norb % 2 == 0
    assert len(target_orbs) % 2 == 0
    mat = np.zeros((norb, norb))
    for i, val in zip(range(len(target_orbs) // 2 - 1), coeffs):
        j, k = target_orbs[i], target_orbs[i + 1]
        mat[j, k] = -val
        mat[k, j] = -val
    for i, val in zip(range(len(target_orbs) // 2, len(target_orbs) - 1), coeffs):
        j, k = target_orbs[i], target_orbs[i + 1]
        mat[j, k] = -val
        mat[k, j] = -val
    coeffs, orbital_rotation = scipy.linalg.eigh(mat)
    return ffsim.apply_num_op_sum_evolution(
        vec,
        coeffs,
        1.0,
        norb=norb,
        nelec=nelec,
        orbital_rotation=orbital_rotation,
        copy=copy,
    )


def _simulate_hop_spinful(
    vec: np.ndarray,
    coeffs: np.ndarray,
    target_orbs: list[int],
    norb: int,
    nelec: tuple[int, int],
    copy: bool,
) -> np.ndarray:
    mat = np.zeros((norb, norb))
    for i, val in zip(range(len(target_orbs) - 1), coeffs):
        j, k = target_orbs[i], target_orbs[i + 1]
        mat[j, k] = -val
        mat[k, j] = -val
    coeffs, orbital_rotation = scipy.linalg.eigh(mat)
    return ffsim.apply_num_op_sum_evolution(
        vec,
        coeffs,
        1.0,
        norb=norb,
        nelec=nelec,
        orbital_rotation=orbital_rotation,
        copy=copy,
    )


def _simulate_interaction(
    vec: np.ndarray,
    interaction: float,
    target_orbs: list[int],
    norb: int,
    nelec: tuple[int, int],
    num_species: int,
    copy: bool,
) -> np.ndarray:
    if num_species == 1:
        return _simulate_interaction_spinless(
            vec=vec,
            interaction=interaction,
            target_orbs=target_orbs,
            norb=norb,
            nelec=nelec,
            copy=copy,
        )
    else:  # num_species == 2
        return _simulate_interaction_spinful(
            vec=vec,
            interaction=interaction,
            target_orbs=target_orbs,
            norb=norb,
            nelec=nelec,
            copy=copy,
        )


def _simulate_interaction_spinless(
    vec: np.ndarray,
    interaction: float,
    target_orbs: list[int],
    norb: int,
    nelec: tuple[int, int],
    copy: bool,
) -> np.ndarray:
    assert len(target_orbs) % 2 == 0
    n_spatial_orbs = len(target_orbs) // 2
    mat = np.zeros((norb, norb))
    mat[target_orbs[:n_spatial_orbs], target_orbs[n_spatial_orbs:]] = interaction
    mat[target_orbs[n_spatial_orbs:], target_orbs[:n_spatial_orbs]] = interaction
    return ffsim.apply_diag_coulomb_evolution(
        vec,
        mat=mat,
        time=1.0,
        norb=norb,
        nelec=nelec,
        copy=copy,
    )


def _simulate_interaction_spinful(
    vec: np.ndarray,
    interaction: float,
    target_orbs: list[int],
    norb: int,
    nelec: tuple[int, int],
    copy: bool,
) -> np.ndarray:
    mat_alpha_beta = np.zeros((norb, norb))
    mat_alpha_beta[target_orbs, target_orbs] = interaction
    return ffsim.apply_diag_coulomb_evolution(
        vec,
        mat=np.zeros((norb, norb)),
        mat_alpha_beta=mat_alpha_beta,
        time=1.0,
        norb=norb,
        nelec=nelec,
        copy=copy,
    )


def _simulate_phase(
    vec: np.ndarray,
    mu: np.ndarray,
    target_orbs: list[int],
    norb: int,
    nelec: tuple[int, int],
    num_species: int,
    copy: bool,
) -> np.ndarray:
    if num_species == 1:
        return _simulate_phase_spinless(
            vec=vec,
            mu=mu,
            target_orbs=target_orbs,
            norb=norb,
            nelec=nelec,
            copy=copy,
        )
    else:  # num_species == 2
        return _simulate_phase_spinful(
            vec=vec,
            mu=mu,
            target_orbs=target_orbs,
            norb=norb,
            nelec=nelec,
            copy=copy,
        )


def _simulate_phase_spinless(
    vec: np.ndarray,
    mu: np.ndarray,
    target_orbs: list[int],
    norb: int,
    nelec: tuple[int, int],
    copy: bool,
) -> np.ndarray:
    assert len(target_orbs) % 2 == 0
    n_spatial_orbs = len(target_orbs) // 2
    coeffs = np.zeros(norb)
    coeffs[target_orbs[:n_spatial_orbs]] = mu
    coeffs[target_orbs[n_spatial_orbs:]] = mu
    return ffsim.apply_num_op_sum_evolution(
        vec, coeffs, time=1.0, norb=norb, nelec=nelec, copy=copy
    )


def _simulate_phase_spinful(
    vec: np.ndarray,
    mu: np.ndarray,
    target_orbs: list[int],
    norb: int,
    nelec: tuple[int, int],
    copy: bool,
) -> np.ndarray:
    coeffs = np.zeros(norb)
    coeffs[target_orbs] = mu
    return ffsim.apply_num_op_sum_evolution(
        vec, coeffs, time=1.0, norb=norb, nelec=nelec, copy=copy
    )


def _simulate_frz(
    vec: np.ndarray,
    phi: np.ndarray,
    target_orbs: list[int],
    norb: int,
    nelec: tuple[int, int],
    num_species: int,
    copy: bool,
) -> np.ndarray:
    if num_species == 1:
        a, b = target_orbs
        vec = ffsim.apply_num_interaction(vec, -phi, a, norb, nelec, copy=copy)
        vec = ffsim.apply_num_interaction(vec, phi, b, norb, nelec, copy=False)
        return vec
    else:  # num_species == 2
        a, b = target_orbs
        spin_a, orb_a = divmod(a, norb)
        spin_b, orb_b = divmod(b, norb)
        spins = (ffsim.Spin.ALPHA, ffsim.Spin.BETA)
        vec = ffsim.apply_num_interaction(vec, -phi, orb_a, norb, nelec, spins[spin_a], copy=copy)
        vec = ffsim.apply_num_interaction(vec, phi, orb_b, norb, nelec, spins[spin_b], copy=False)
        return vec


def _fermionic_op_to_fermion_operator(  # pylint: disable=invalid-name
    op: FermionicOp, target_orbs: list[int]
) -> ffsim.FermionOperator:
    """Convert a Qiskit Nature FermionicOp to an ffsim FermionOperator."""
    norb_small = len(target_orbs)
    coeffs = {}
    for term, coeff in op.terms():
        fermion_actions = []
        for action_str, index in term:
            action = action_str == "+"
            spin, orb = divmod(index, norb_small)
            fermion_actions.append((action, bool(spin), target_orbs[orb]))
        coeffs[tuple(fermion_actions)] = coeff
    return ffsim.FermionOperator(coeffs)
