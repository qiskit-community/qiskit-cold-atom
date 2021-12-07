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

"""Backend to emulate fermionic tweezer experiments."""

from qiskit_cold_atom.fermions.fermion_simulator_backend import FermionSimulator


class FermionicTweezerSimulator(FermionSimulator):
    """Emulator backend of a fermionic device with n tweezer sites that uses two spin species.

    This backend will define a configuration with a set of supported gates and a line coupling map.
    At instantiation users can chose the length of the coupling map which by default is four sites.
    This backend is intended to be a realistic representation of a ferminoic tweezer experiment.
    For a general fermionic simulator backend, use the FermionSimulator base class.
    """

    def __init__(self, n_tweezers: int = 4, provider=None):
        """Create a new fermionic tweezer simulator backend.

        Args:
            n_tweezers: The number of optical tweezers.
            provider: The provider to which the backend may be added.
        """

        # define coupling maps for the gates
        sites = n_tweezers
        self._neighbouring_sites_couplings = [
            list(range(i, i + size)) + list(range(i + sites, i + sites + size))
            for size in range(2, sites + 1)
            for i in range(sites + 1 - size)
        ]
        self._global_site_couplings = [list(range(2 * sites))]
        self._single_site_couplings = [[i, i + sites] for i in range(sites)]

        configuration = {
            "backend_name": "fermionic_tweezer_simulator",
            "cold_atom_type": "fermion",
            "backend_version": "0.0.1",
            "simulator": True,
            "local": True,
            "coupling_map": None,
            "description": "Mock backend of a fermionic tweezer hardware. The first half of wires in a "
            "circuit denote the occupations of the spin-up fermions and the last half "
            "of wires denote the spin-down fermions",
            "basis_gates": [
                "hop",
                "int",
                "phase",
                "FH",
                "fer_rx",
                "fer_ry",
                "fer_rz",
                "load",
            ],
            "num_species": 2,
            "memory": True,
            "n_qubits": 2 * sites,
            "conditional": False,
            "max_shots": 1e6,
            "max_experiments": 10,
            "open_pulse": False,
            "gates": [
                {
                    "name": "hop",
                    "parameters": ["j_i"],
                    "qasm_def": "{}",
                    "description": "hopping of atoms to neighboring tweezers",
                    "coupling_map": self._neighbouring_sites_couplings,
                },
                {
                    "name": "int",
                    "parameters": ["u"],
                    "qasm_def": "{}",
                    "description": "on-site interaction of atoms of opposite spin state",
                    "coupling_map": self._global_site_couplings,
                },
                {
                    "name": "phase",
                    "parameters": ["mu_i"],
                    "qasm_def": "{}",
                    "description": "Applying a local phase to tweezers through an external potential",
                    "coupling_map": self._single_site_couplings,
                },
                {
                    "name": "fer_rx",
                    "parameters": ["phi"],
                    "qasm_def": "{}",
                    "description": "x-rotation between the spin-up and spin-down state at one "
                    "tweezer site",
                    "coupling_map": self._single_site_couplings,
                },
                {
                    "name": "fer_ry",
                    "parameters": ["phi"],
                    "qasm_def": "{}",
                    "description": "y-rotation between the spin-up and spin-down state at one "
                    "tweezer site",
                    "coupling_map": self._single_site_couplings,
                },
                {
                    "name": "fer_rz",
                    "parameters": ["phi"],
                    "qasm_def": "{}",
                    "description": "z-rotation between the spin-up and spin-down state at one "
                    "tweezer site",
                    "coupling_map": self._single_site_couplings,
                },
            ],
            "supported_instructions": [
                "load",
                "measure",
                "barrier",
                "hop",
                "int",
                "phase",
                "FH",
                "fer_rx",
                "fer_ry",
                "fer_rz",
            ],
        }

        super().__init__(config_dict=configuration, provider=provider)
