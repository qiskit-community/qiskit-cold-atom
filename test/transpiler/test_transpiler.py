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

"""Testing module for transpiling."""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.test import QiskitTestCase

from qiskit_cold_atom.transpiler import Optimize1SpinGates


class TestSpinTranspilation(QiskitTestCase):
    """Test class for spin-based transpilation."""

    def test_optimize_1s_gates(self):
        """Test the single-spin gate transpilation."""

        circ = QuantumCircuit(1)
        circ.rlx(np.pi / 2, 0)
        circ.rlx(np.pi / 2, 0)
        circ.rlx(np.pi / 2, 0)
        circ.rly(np.pi / 2, 0)
        circ.rly(np.pi / 2, 0)
        circ.rlz2(np.pi / 4, 0)
        circ.rlz2(np.pi / 4, 0)
        circ.rlz2(np.pi / 4, 0)
        circ.rlx(np.pi / 2, 0)

        pass_manager = PassManager(Optimize1SpinGates())

        circ_new = pass_manager.run(circ)

        self.assertEqual(circ_new.count_ops()["rLx"], 2)
        self.assertEqual(circ_new.count_ops()["rLy"], 1)
        self.assertEqual(circ_new.count_ops()["rLz2"], 1)

        self.assertTrue(np.allclose(circ_new.data[0][0].params[0], 3 * np.pi / 2))
        self.assertTrue(np.allclose(circ_new.data[1][0].params[0], np.pi))
        self.assertTrue(np.allclose(circ_new.data[2][0].params[0], 3 * np.pi / 4))
        self.assertTrue(np.allclose(circ_new.data[3][0].params[0], np.pi / 2))

    def test_optimize_1s_gates_multi_spin(self):
        """Test the single-spin gate transpilation."""

        circ = QuantumCircuit(2)
        circ.rlx(np.pi / 3, 0)
        circ.rlx(np.pi / 3, 0)
        circ.rlx(np.pi / 3, 0)
        circ.rly(np.pi / 4, 1)
        circ.rly(np.pi / 4, 1)

        pass_manager = PassManager(Optimize1SpinGates())

        circ_new = pass_manager.run(circ)

        self.assertEqual(circ_new.count_ops()["rLx"], 1)
        self.assertEqual(circ_new.count_ops()["rLy"], 1)

        self.assertTrue(np.allclose(circ_new.data[0][0].params[0], np.pi))
        self.assertTrue(np.allclose(circ_new.data[1][0].params[0], np.pi / 2))
