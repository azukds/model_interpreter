Contributing
============

Thanks for your interest in contributing to this package! No contribution is too small! We're hoping it can be made even better through community contributions.

Requests and feedback
---------------------

For any bugs, issues or feature requests please open an `issue <https://github.com/lvgig/model_interpreter/issues>`_ on the project.

Requirements for contributions
------------------------------

We have some general requirements for all contributions. This is to ensure consistency with the existing codebase.

Set up development environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For External contributors, first create your own fork of this repo.

Then clone the fork (or this repository if internal);

   .. code::

     git clone https://github.com/lvgig/model_interpreter.git
     cd model_interpreter

Then install tubular and dependencies for development;

   .. code::

     pip install . -r requirements-dev.txt

We use `pre-commit <https://pre-commit.com/>`_ for this project which is configured to check that code is formatted with `black <https://black.readthedocs.io/en/stable/>`_

To configure ``pre-commit`` for your local repository run the following;

   .. code::

     pre-commit install

If working in a codespace the dev requirements and precommit will be installed automatically in the dev container.

If you are building the documentation locally you will need the `docs/requirements.txt <https://github.com/lvgig/model_interpreter/blob/main/docs/requirements.txt>`_.

General
^^^^^^^

- Please try and keep each pull request to one change or feature only
- Make sure to update the `changelog <https://github.com/lvgig/model_interpreter/blob/main/CHANGELOG.rst>`_ with details of your change

Code formatting
^^^^^^^^^^^^^^^

We use `black <https://black.readthedocs.io/en/stable/>`_ to format our code and follow `pep8 <https://www.python.org/dev/peps/pep-0008/>`_ conventions. 

As mentioned above we use ``pre-commit`` which streamlines checking that code has been formatted correctly.

CI
^^

Make sure that pull requests pass our `CI <https://github.com/lvgig/model_interpreter/actions>`_. It includes checks that;

- code is formatted with `black <https://black.readthedocs.io/en/stable/>`_
- `flake8 <https://flake8.pycqa.org/en/latest/>`_ passes
- the tests for the project pass, with a minimum of 80% branch coverage
- `bandit <https://bandit.readthedocs.io/en/latest/>`_ passes

Tests
^^^^^

We use `pytest <https://docs.pytest.org/en/stable/>`_ as our testing framework.

All existing tests must pass and new functionality must be tested. We aim for 100% coverage on new features that are added to the package.

We also make use of the `test-aide <https://github.com/lvgig/test-aide>`_ package to make mocking easier and to help with generating data when `parametrizing <https://docs.pytest.org/en/6.2.x/parametrize.html>`_ tests.

Docstrings
^^^^^^^^^^

We follow the `numpy <https://numpydoc.readthedocs.io/en/latest/format.html>`_ docstring style guide.

Docstrings need to be updated for the relevant changes and docstrings need to be added for new methods.
