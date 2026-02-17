Tutorial: Stable Diffusion
==========================

.. rst-class:: lead

   A tutorial on how to define Stable Diffusion 1.5 as a base model in *diffusiongym*.

Data type
---------

For images, we can simply use tensors as the datatype. *diffusiongym* already has a wrapper for this in
``DDTensor``. See :ref:`diffusiongym/types.py` for implementation details.

Scheduler
---------

Stable Diffusion is trained as a DDPM through a very specific noise schedule from times
:math:`T` to :math:`0`, where:

.. math::

   \mathbf{x}_{t} = \sqrt{\bar{\gamma}_t} \mathbf{x}_t + \sqrt{1-\bar{\gamma}_t} \mathbf{\epsilon}, \quad \mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})

This is equivalent to the following flow matching schedule from times :math:`0` to :math:`1`:

.. math::

   \alpha_t & = \sqrt{\bar{\gamma}_t} \\
   \beta_t & = \sqrt{1 - \bar{\gamma}_t} \\
   \dot{\alpha}_t & = \frac{\dot{\bar{\gamma}}_t}{2\sqrt{\bar{\gamma}_t}} \\
   \dot{\beta}_t & = -\frac{\dot{\bar{\gamma}}_t}{2\sqrt{1 - \bar{\gamma}_t}}

To convert time conventions, we use the following:

.. math::

   \bar{\gamma}_t = \bar{\gamma} \lfloor T(1-t) \rfloor

And we compute its time derivative by finite differences:

.. math::

   \dot{\bar{\gamma}}_t = T \cdot \left( \bar{\gamma}\lfloor T(1-t) - 1 \rfloor - \bar{\gamma}\lfloor T(1-t) \rfloor \right)

*diffusiongym* implements this through ``DiffusionScheduler`` which takes the :math:`\bar{\gamma}`
noise schedule as input.

Base model
----------

To obtain the base model, we make use of the ``diffusers`` library by Hugging Face. We can easily
obtain the base model through their API:

.. code-block:: python

   from diffusers import StableDiffusionPipeline
   from diffusiongym import BaseModel, DDTensor, base_model_registry

   @base_model_registry.register("images/stable_diffusion")
   class StableDiffusionBaseModel(BaseModel[DDTensor]):
      output_type = "epsilon"

      def __init__(self, device):
         super().__init__(device)
         self.pipe = StableDiffusionPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-v1-5",
            device=device,
         )

We then set the scheduler as follows:

.. code-block:: python

   from diffusiongym import DiffusionScheduler

   # In StableDiffusionBaseModel.__init__
   alphas_cumprod = self.pipe.scheduler.alphas_cumprod.to(device)
   self._scheduler = DiffusionScheduler(alphas_cumprod)

   # In StableDiffusionBaseModel
   @property
   def scheduler(self):
      return self._scheduler

Sample initial distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stable Diffusion has :math:`4\times 64\times 64` latent dimensions. We can sample from the initial
distribution by sampling from the standard normal distribution. Furthermore, we need a prompt to
condition on. For now, we will use a constant prompt, but this could also be randomly selected from a
dataset.

.. code-block:: python

   # In StableDiffusionBaseModel
   def sample_p0(self, n: int, **kwargs):
      latents = torch.randn(n, 4, 64, 64, device=self.device)
      prompt = kwargs.get("prompt", "A photo of a cat")

      if isinstance(prompt, str):
         prompt = [prompt] * n

      return DDTensor(latents), { "prompt": prompt }

Preprocessing
^^^^^^^^^^^^^

In order for the U-net base model to make predictions, we need to encode the prompt:

.. code-block:: python

   # In StableDiffusionBaseModel
   def preprocess(self, x: DDTensor, **kwargs):
      prompt_embeds, _ = self.pipe.encode_prompt(prompt, self.device, 1, False)
      return x, { "encoder_hidden_states": prompt_embeds }

Forward pass
^^^^^^^^^^^^

Now the forward pass involves passing the noisy latents, timestep, and the encoded prompt to the
U-net model, which predicts the noise:

.. code-block:: python

   # In StableDiffusionBaseModel
   def forward(self, x: DDTensor, t: torch.Tensor, **kwargs):
      y = x.data
      k = self.scheduler.model_input(t)
      return DDTensor(self.pipe.unet(y, k, kwargs["encoder_hidden_states"]).sample)

This can additionally be altered by adding classifier-free guidance, as in
:ref:`diffusiongym/images/base_models/stable_diffusion.py`.

Postprocessing
^^^^^^^^^^^^^^

Lastly, we need to decode the latents back into images through the VAE decoder:

.. code-block:: python

   # In StableDiffusionBaseModel
   def postprocess(self, x: DDTensor, **kwargs):
      y = x.data / self.pipe.vae.config.scaling_factor
      images = self.pipe.vae.decode(y).sample
      images = (images + 1) / 2
      images = images.clamp(0, 1)
      return DDTensor(images)

Reward function
---------------

To finish the setup, we can define a reward function. Here, we will use a simple reward that
rewards images that are more "red" by summing up the red channel pixel values. And since all images
are valid, we output all ones for the second return value:

.. code-block:: python

   from diffusiongym import Reward, DDTensor

   @reward_registry.register("images/red")
   class RednessReward(Reward[DDTensor]):
      def forward(self, x: DDTensor, **kwargs):
         red_channel = x.data[:, 0, :, :]
         return red_channel.mean(dim=(1, 2)).cpu(), torch.ones(x.shape[0])

Environment
-----------

Now we can combine the base model and reward function into a *diffusiongym* environment:

.. code-block:: python

   import diffusiongym

   env = diffusiongym.make(
      "images/stable_diffusion",
      "images/red",
      discretization_steps=50,
      reward_scale=100,
      device=device,
   )
