# Group feature importance

This repo contains the implementation to compute feature importance of correlated features.

## Installation

### Installing with `pip`

TODO

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
