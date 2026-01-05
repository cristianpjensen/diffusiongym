Flow Gym
========

.. rst-class:: lead

   Library for reward adaptation of any pre-trained flow model on any data modality.

.. image:: _static/teaser.gif
   :alt: Flow Gym Teaser
   :align: center

Installation
------------

In order to install *flowgym*, execute the following command:

.. code-block:: bash

   pip install flowgym

High-level overview
-------------------

Diffusion and flow models are largely agnostic to their data modality. They only require that the
underlying data type supports a small set of operations. Building on this idea, *flowgym* is
designed to be fully modular. You only need to provide the following:

- Data type ``YourDataType`` that implements ``FlowProtocol``, which defines some functions necessary for interacting with it as a flow model.
- Base model ``BaseModel[YourDataType]``, which defines the scheduler, how to sample :math:`p_0`, how to compute the forward pass, and how to preprocess and postprocess data.
- Reward function ``Reward[YourDataType]``.

Once these are defined, you can sample from the flow model and apply reward adaptation methods, such
as Value Matching.

Table of contents
-----------------

.. toctree::
   :caption: How To
   :titlesonly:

   math
   quickstart
   registries
   policies
   stable_diffusion

.. toctree::
   :caption: API Reference
   :titlesonly:

   api/environments
   api/base_models
   api/schedulers
   api/rewards
   api/types
   api/images
   api/molecules
