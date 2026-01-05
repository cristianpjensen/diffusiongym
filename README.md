# Flow Gym

<div align="center">
  <img src="docs/_static/teaser.gif" width="100%" />
</div>

<p align="center">
<a href="https://github.com/cristianpjensen/flow-gym/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/cristianpjensen/flowgym"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Code style: ruff" src="https://img.shields.io/badge/code%20style-ruff-000000.svg"></a>
</p>

`flowgym` is a library for reward adaptation of any pre-trained flow model on any data modality.

## Installation

In order to install *flowgym*, execute the following command:
```bash
pip install flowgym
```

*flowgym* requires PyTorch 2.3.1, and there may be other hard dependencies. Please open an issue if
installation fails through the above command.

## High-level overview

Diffusion and flow models are largely agnostic to their data modality. They only require that the underlying data type supports a small set of operations. Building on this idea, *flowgym* is designed to be fully modular. You only need to provide the following:
 * Data type `YourDataType` that implements `FlowProtocol`, which defines some functions necessary for interacting with it as a flow model.
 * Base model `BaseModel[YourDataType]`, which defines the scheduler, how to sample $p_0$, how to compute the forward pass, and how to preprocess and postprocess data.
 * Reward function `Reward[YourDataType]`.

Once these are defined, you can sample from the flow model and apply reward adaptation methods, such as Value Matching.

## Documentation

Much more information can be found in [the documentation](https://cristianpjensen.github.io/flowgym/), including tutorials and API references.
