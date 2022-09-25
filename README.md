# Group feature importance

<p align="center">
    <a href="https://github.com/BorealisAI/group-feature-importance/actions">
      <img alt="CI" src="https://github.com/BorealisAI/group-feature-importance/workflows/CI/badge.svg?event=push&branch=main">
    </a>
    <a href="https://pypi.org/project/groufi/">
      <img alt="Python" src="https://img.shields.io/pypi/pyversions/groufi.svg">
    </a>
    <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
      <img alt="Attribution-NonCommercial-ShareAlike 4.0 International" src="https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg">
    </a>
    <a href="https://codecov.io/gh/durandtibo/group-feature-importance">
      <img alt="Codecov" src="https://codecov.io/gh/durandtibo/group-feature-importance/branch/main/graph/badge.svg?token=IRVV3WC71O">
    </a>
    <a href="https://github.com/psf/black">
     <img  alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <br/>
    <a href="https://twitter.com/intent/follow?screen_name=BorealisAI">
        <img src="https://img.shields.io/twitter/follow/BorealisAI?style=social&logo=twitter" alt="follow on Twitter">
    </a>
    <br/>
</p>


This repo contains the implementation to compute feature importance of correlated features.

## Examples

Some examples are available in [`examples`](examples)

## Installation

### Installing with `pip`

This repository is tested on Python 3.9, and Linux systems.
It is recommended to install in a virtual environment to keep your system in order.
The following command installs the latest version of the library:

```shell
pip install groufi
```

### Installing from source

To install `groufi` from source, you can follow the steps below. First, you will need to
install [`poetry`](https://python-poetry.org/docs/master/). `poetry` is used to manage and install the dependencies.
If `poetry` is already installed on your machine, you can skip this step. There are several ways to install `poetry` so
you can use the one that you prefer. You can check the `poetry` installation by running the following command:

```shell
poetry --version
```

Then, you can clone the git repository:

```shell
git clone git@github.com:BorealisAI/group-feature-importance.git
```

Then, it is recommended to create a Python 3.8+ virtual environment. This step is optional so you can skip it. To create
a virtual environment, you can use the following command:

```shell
make conda
```

It automatically creates a conda virtual environment. When the virtual environment is created, you can activate it with
the following command:

```shell
conda activate groufi
```

This example uses `conda` to create a virtual environment, but you can use other tools or configurations. Then, you
should install the required package to use `groufi` with the following command:

```shell
make install
```

This command will install all the required packages. You can also use this command to update the required packages. This
command will check if there is a more recent package available and will install it. Finally, you can test the
installation with the following command:

```shell
make test
```

## License

This repository is released under the Attribution-NonCommercial-ShareAlike 4.0 International license as found in
the [LICENSE](LICENSE) file.
