# Contributing

**We appreciate all kinds of help, so thank you!**

First please read the overall project contributing guidelines. These are
included in the Qiskit documentation here:

https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md

## Contributing to Qiskit Cold Atom

If you've noticed a bug or have a feature request, we encourage to open an issue in
[this repo's issues tracker](https://github.com/qiskit-partners/qiskit-cold-atom/issues), 
whether you plan to address it yourself or not.

If you intend to contribute code, please still start the contribution process by opening a new issue or 
making a comment on an existing issue briefly explaining what you intend to address and how. 
This helps us understand your intent/approach and provide support and commentary before you take the 
time to actually write code, and ensures multiple people aren't accidentally working on the same thing.

### Project Code Style.

Code in Qiskit Cold Atom should conform to PEP8 and style/lint checks are run to validate
this.  Line length must be limited to no more than 100 characters. Docstrings
should be written using the Google docstring format.

### Making a pull request

When you're ready to make a pull request, please make sure the following is true:

1. The code matches the project's code style
2. The documentation, _including any docstrings for changed methods_, has been updated
3. If appropriate for your change, that new tests have been added to address any new functionality, or that existing tests have been updated as appropriate
4. All of the tests (new and old) still pass!
5. You have added notes in the pull request that explains what has changed and links to the relevant issues in the issues tracker


### Building the documentation

The documentation for the Python SDK is auto-generated from Python
docstrings using [Sphinx](http://www.sphinx-doc.org. Please follow [Google's Python Style
Guide](https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments)
for docstrings. A good example of the style can also be found with
[Sphinx's napolean converter
documentation](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
To build the documentation locally navigate to `.\docs\` and run

```bash
sphinx-build -M html . _build
```
which creates the documentation at `.\docs\_build\html\index.html` .

### Running the tests

This package uses the [pytest](https://docs.pytest.org/en/stable/) test runner.

To use pytest directly, just run:

```bash
pytest [pytest-args]
```
