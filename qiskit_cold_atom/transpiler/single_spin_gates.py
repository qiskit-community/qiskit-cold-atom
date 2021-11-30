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

"""A transpiler pass to simplify single-qubit gates."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from qiskit_cold_atom.spins.spins_gate_library import (
    RLXGate,
    RLYGate,
    RLZGate,
    RLZ2Gate,
)


class Optimize1SpinGates(TransformationPass):
    """Simplify single-spin spin gates.

    The simplification of these gates is done by adding the angle of identical and
    consecutive gates together. For example, the circuit

    .. parsed-literal::

                ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
        spin_0: ┤ Rlx(π/2) ├┤ Rlx(π/2) ├┤ Rlx(π/2) ├┤ Rly(π/2) ├┤ Rly(π/2) ├┤ Rlx(π/2) ├
                └──────────┘└──────────┘└──────────┘└──────────┘└──────────┘└──────────┘

    will be transpiled to

    .. parsed-literal::

                ┌───────────┐┌────────┐┌──────────┐
        spin_0: ┤ Rlx(3π/2) ├┤ Rly(π) ├┤ Rlx(π/2) ├
                └───────────┘└────────┘└──────────┘
    """

    def __init__(self):
        super().__init__()
        self._gate_names = ["rLx", "rLy", "rLz", "rLz2"]

    def run(self, dag):
        """Run the Optimize1qSpinGates pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """

        for gate_name in self._gate_names:
            runs = dag.collect_runs(gate_name)

            for run in runs:
                if len(run) == 1:
                    continue

                # Sum the angles of the same single-qubit gates.
                total_angle = 0.0
                for dag_node in run:
                    total_angle += dag_node.op.params[0]

                if gate_name == "rLx":
                    new_op = RLXGate(total_angle)
                elif gate_name == "rLy":
                    new_op = RLYGate(total_angle)
                elif gate_name == "rLz":
                    new_op = RLZGate(total_angle)
                elif gate_name == "rLz2":
                    new_op = RLZ2Gate(total_angle)
                else:
                    raise TranspilerError(
                        f"Could not use the basis {self._gate_names}."
                    )

                dag.substitute_node(run[0], new_op, inplace=True)

                # Delete the remaining nodes
                for current_node in run[1:]:
                    dag.remove_op_node(current_node)

        return dag
