# Qiskit Cold Atom 

<img src="docs/images/qiskit_cold_atom_logo_with_text.svg" alt="Qiskit cold atom logo" style="height: 70px; width:70px;"/>

**Qiskit** is an open-source SDK for working with quantum computers at the level of circuits, algorithms, and application modules.

This project builds on this functionality to describe programmable quantum simulators of trapped cold atoms 
in a gate- and circuit-based framework. This includes a provider that allows access to cold atomic
quantum devices located at Heidelberg University.

Traditionally, each wire in a quantum circuit represents one qubit as the fundamental unit of information processing. 
Here, we extend this concept and allow wires to represent individual internal states of trapped cold atoms. 
This currently covers two settings, one for fermionic modes and one for spin modes, 
demonstrating that a broad range of hardware can be accommodated in Qiskit.

We encourage new users to familiarize themselves with the basic functionalities through the tutorial-style python notebooks in `/docs/tutorials`. 

## Installation

To install Qiskit Cold Atom from source, clone this repository and install from the project root using pip:
```bash
pip install -e .
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
from the qubit circuits typically employed in Qiskit, because their systems are described by different
Hilbert spaces. 
To see how to define and run gates through quantum circuits in this setting, please refer to the tutorials (in `/docs/tutorials`).

## Documentation

To build the API reference locally, run:

```bash
pip install -r requirements-dev.txt
make -C docs html
open docs/_build/html/index.html
```

## License

[Apache License 2.0].

[Apache License 2.0]: https://github.com/qiskit-community/qiskit-cold-atom/blob/master/LICENSE.txt