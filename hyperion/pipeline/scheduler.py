"""
Hyperion Pipeline: Flow Match Scheduler
=======================================
Implements rectified-flow / flow-matching scheduling for MM-DiT backbones
(FLUX.2-Klein-9B and FLUX.1-dev).

Provides a pure-PyTorch implementation of the Euler flow-match scheduler,
enabling inference and testing without requiring third-party scheduler packages.
"""

from typing import Optional, Union
import math
import torch


class FlowMatchScheduler:
    """
    Rectified Flow / Flow Matching Scheduler.

    Drives the denoising ODE:
        dx_t / dt = v_t,  t in [1000, 0] (or [1, 0])
    with dynamic resolution-dependent time-shift.
    """

    def __init__(
        self,
        shift: float = 3.0,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        extra_one_step: bool = True,
    ):
        self.shift = shift
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.extra_one_step = extra_one_step
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        dynamic_shift_len: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        Computes timesteps and sigmas schedule.

        Args:
            num_inference_steps: Number of denoising steps (default: 25).
            dynamic_shift_len: Latent token sequence length (H*W) for dynamic shift.
            device: Computation device.
        """
        # Linear sigma schedule from 1.0 down to 0.0
        sigmas = torch.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1)

        # Apply FLUX resolution-dependent time-shift
        shift_val = self.shift
        if dynamic_shift_len is not None and dynamic_shift_len > 0:
            # Shift scales with token length
            base_seq_len = 256.0
            scale = math.sqrt(float(dynamic_shift_len) / base_seq_len)
            shift_val = math.exp(math.log(max(shift_val, 1.01)) * max(scale, 0.5))

        if shift_val != 1.0:
            # sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
            sigmas = (sigmas * shift_val) / (1.0 + (shift_val - 1.0) * sigmas)

        self.sigmas = sigmas.to(device)
        # Timesteps in [1000, 0]
        self.timesteps = (sigmas[:-1] * 1000.0).to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Advances the latent state by one Euler step:
            x_{next} = x_{curr} + v * (sigma_{next} - sigma_{curr})

        Args:
            model_output: Model predicted velocity field v_t.
            timestep: Current timestep.
            sample: Current latent tensor x_t.

        Returns:
            Next latent tensor x_{t - dt}.
        """
        t_val = float(timestep.item()) if hasattr(timestep, "item") else float(timestep)

        # Find closest timestep index
        if self.timesteps is not None:
            diffs = (self.timesteps - t_val).abs()
            step_idx = int(torch.argmin(diffs).item())
            if step_idx < len(self.sigmas) - 1:
                sigma_curr = self.sigmas[step_idx]
                sigma_next = self.sigmas[step_idx + 1]
                dt = sigma_next - sigma_curr
            else:
                dt = torch.tensor(-1.0 / len(self.timesteps), device=sample.device, dtype=sample.dtype)
        else:
            dt = torch.tensor(-1.0 / 25.0, device=sample.device, dtype=sample.dtype)

        dt = dt.to(device=sample.device, dtype=sample.dtype)
        return sample + model_output * dt

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """
        Adds flow-matching noise to clean sample:
            x_t = (1 - sigma) * x_clean + sigma * noise

        Args:
            original_samples: Clean latent tensor x_1.
            noise: Gaussian noise tensor x_0.
            timestep: Timestep tensor or scalar in [1000, 0].

        Returns:
            Noised latent tensor x_t.
        """
        t_val = float(timestep.item()) if hasattr(timestep, "item") else float(timestep)
        sigma = max(0.0, min(1.0, t_val / 1000.0))
        orig_dtype = original_samples.dtype
        res = (1.0 - sigma) * original_samples.float() + sigma * noise.float()
        return res.to(orig_dtype)
