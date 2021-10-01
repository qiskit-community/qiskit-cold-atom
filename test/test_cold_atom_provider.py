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

"""Cold atom provider tests"""

import os
from qiskit.test import QiskitTestCase
from qiskit_cold_atom.providers import ColdAtomProvider


class TestHeidelbergProvider(QiskitTestCase):
    """HeidelbergColdAtomProvider tests."""

    def setUp(self):
        super().setUp()

        # create directory for credentials if it doesn't already exist
        self.path = os.path.join(os.path.expanduser("~"), ".qiskit")
        self.path_exists = os.path.isdir(self.path)
        if not self.path_exists:
            os.mkdir(self.path)

        # Store some credentials
        self.username = "test_user"
        self.token = "test_token"
        self.url = "http://localhost:9000/shots"
        self.filename = os.path.join(self.path, "cold_atom_provider_test")

    def test_credential_management(self):
        """Test the management of locally stored credential data"""
        with self.subTest("test save account"):
            ColdAtomProvider.save_account(
                url=self.url,
                username=self.username,
                token=self.token,
                filename=self.filename,
            )
            stored_credentials = ColdAtomProvider.stored_account(self.filename)
            target = {
                "urls": [self.url],
                "username": self.username,
                "token": self.token,
            }
            self.assertTrue(stored_credentials == target)

        with self.subTest("test overwrite warning"):
            with self.assertWarns(UserWarning):
                ColdAtomProvider.save_account(
                    url=self.url,
                    username=self.username,
                    token=self.token,
                    filename=self.filename,
                )

        with self.subTest("test add url"):
            ColdAtomProvider.add_url("second_url", filename=self.filename)
            stored_credentials = ColdAtomProvider.stored_account(self.filename)
            target = {
                "urls": [self.url, "second_url"],
                "username": self.username,
                "token": self.token,
            }
            self.assertTrue(stored_credentials == target)

        with self.subTest("test delete account"):
            with self.assertWarns(UserWarning):
                ColdAtomProvider.delete_account(
                    filename=os.path.join(
                        os.path.expanduser("~"), ".qiskit", "wrong_filename"
                    )
                )
            ColdAtomProvider.delete_account(filename=self.filename)
            stored_credentials = ColdAtomProvider.stored_account(self.filename)
            self.assertTrue(stored_credentials == {})

    def test_provider_initialization(self):
        """Test the initialization of a cold atom provider from invalid credentials"""
        ColdAtomProvider.save_account(
            url=self.url,
            username=self.username,
            token=self.token,
            filename=self.filename,
        )
        with self.subTest("test load account"):
            # test that a warning is raised when initializing with invalid credentials
            with self.assertWarns(UserWarning):
                provider = ColdAtomProvider.load_account(filename=self.filename)
                target = {
                    "urls": [self.url],
                    "username": self.username,
                    "token": self.token,
                }
                self.assertTrue(provider.active_account() == target)

    def tearDown(self):
        super().tearDown()
        os.remove(self.filename)
        if not self.path_exists:
            os.rmdir(self.path)
