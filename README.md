# Qiskit Cold Atom 

<img src="docs/images/qiskit_cold_atom_logo_with_text.svg" alt="Qiskit cold atom logo" style="height: 228px; width:450px;"/>

**Qiskit** is an open-source SDK for working with quantum computers at the level of circuits, algorithms, and application modules.

This project builds on this functionality to describe programmable quantum simulators of trapped cold atoms 
in a gate- and circuit-based framework. This includes a provider that allows access to cold atomic
quantum devices located at Heidelberg University.

Traditionally, each wire in a quantum circuit represents one qubit as the fundamental unit of information processing. 
Here, we extend this concept and allow wires to represent individual internal states of trapped cold atoms. 
This currently covers two settings, one for fermionic modes and one for spin modes, 
demonstrating that a broad range of hardware can be accommodated in Qiskit.

We encourage new users to familiarize themselves with the basic functionalities through the tutorial-style python notebooks in [`docs/tutorials`](docs/tutorials). 
These require an environment to execute `.ipynb` notebooks such as jupyter lab. 

## Installation

To install Qiskit Cold Atom from source and further develop the package, clone this repository and install from the project root using pip:
```bash
pip install -e .
```
To use the repository you can also install using either
```bash
pip install git+https://github.com/qiskit-community/qiskit-cold-atom.git
```
or Pypi
```bash
pip install qiskit-cold-atom
```
To install Qiskit Cold Atom with the [ffsim fermion simulator backend](#cold-atomic-circuits) (not supported on Windows), specify the `ffsim` extra in the `pip` install command, e.g.
```bash
pip install "qiskit-cold-atom[ffsim]"
```

## Setting up the Cold Atom Provider 
Qiskit Cold Atom includes local simulator backends to explore the cold atomic hardware. In order to access
remote device backends, you will need valid user credentials. 
To this end, the user has to specify the `url` of the desired backend and a valid `username` and `token` as a password, 
which can be saved as an account: 

```python
from qiskit_cold_atom.providers import ColdAtomProvider

# save an account to disk
ColdAtomProvider.save_account(urls = ["url_backend_1", "url_backend_2"], username="my_name", token="my_password") 
```

Loading the account instantiates the provider from which the included backends can be accessed. 

```python
# load the stored account
provider = ColdAtomProvider.load_account()

# get available backends
print(provider.backends())

# Example: Get a simulator backend
spin_simulator_backend = provider.get_backend("collective_spin_simulator")
```

## Cold atomic circuits

The circuits that can be run on the cold atomic hardware explored in this project use different gates
from the circuits typically employed in Qiskit, because these hardware are not built from qubits,
but from fermions or spins.
Qiskit Cold Atom includes basic simulators for both the fermion and spin settings that can be used
to simulate small circuits. See [Introduction & Fermionic Circuits](docs/tutorials/01_introduction_and_fermionic_circuits.ipynb)
and [Spin circuits](docs/tutorials/02_spin_circuits.ipynb) for tutorials on how to define and run gates through
quantum circuits in these settings.

Qiskit Cold Atom also includes a high-performance simulator for fermionic circuits based on
[ffsim](https://github.com/qiskit-community/ffsim), which can handle much larger circuits than the basic simulator mentioned before. The ffsim simulator is not supported on Windows, and in order
for it to be available, Qiskit Cold Atom must be installed with the `ffsim` extra, e.g.
```bash
pip install "qiskit-cold-atom[ffsim]"
```

## Documentation

The documentation can be found as Github pages here [https://qiskit-community.github.io/qiskit-cold-atom/](https://qiskit-community.github.io/qiskit-cold-atom/).
To build the API reference locally, run:

```bash
pip install -r requirements-dev.txt
make -C docs html
open docs/_build/html/index.html
```

## Tests
Test are located in the `test` folder. All contributions should come with test files that are named `test_*.py` which test the new functionalities. 
To execute the test suite locally, run
```bash
python -m unittest
```
from the project root. 

## License

[Apache License 2.0].

[Apache License 2.0]: https://github.com/qiskit-community/qiskit-cold-atom/blob/master/LICENSE.txt
