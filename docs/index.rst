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

If you want access to pre-trained image or molecular generation models, specify them as options:

.. code-block:: bash

   pip install flowgym[images]
   pip install flowgym[molecules]

High-level overview
-------------------

Diffusion and flow models are largely agnostic to their data modality. They only require that the
underlying data type supports a small set of operations. Building on this idea, *flowgym* is
designed to be fully modular. You only need to provide the following:

- Data type that implements ``DataProtocol``, which defines basic arithmetic operations, factory methods, and gradient methods.
- Base model ``BaseModel[DataType]``, which defines the scheduler, how to sample :math:`p_0`, how to compute the forward pass, and how to preprocess and postprocess data.
- Reward function ``Reward[DataType]``.

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

.. toctree::
   :caption: Optional
   :titlesonly:

   api/images
   api/molecules
