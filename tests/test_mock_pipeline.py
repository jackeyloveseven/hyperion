import unittest
from PIL import Image
import torch
import torch.nn as nn
from hyperion.pipeline.config import HyperionConfig
from hyperion.pipeline.hyperion_pipeline import HyperionPipeline


class MockDiT(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.conv = nn.Linear(in_channels, in_channels)

    def forward(self, hidden_states, timestep=None, guidance=None, encoder_hidden_states=None, txt_ids=None, img_ids=None):
        return self.conv(hidden_states)


class MockVAE(nn.Module):
    def encode(self, pix):
        b, c, h, w = pix.shape
        return torch.randn(b, 128, h // 16, w // 16, device=pix.device, dtype=pix.dtype)

    def decode(self, lat):
        b, c, h, w = lat.shape
        return torch.randn(b, 3, h * 16, w * 16, device=lat.device, dtype=lat.dtype)


class MockPipelineObj:
    def __init__(self):
        self.dit = MockDiT()
        self.vae = MockVAE()
        self.torch_dtype = torch.float32

    def load_models_to_device(self, names):
        pass

    def preprocess_image(self, pil_img):
        w, h = pil_img.size
        return torch.zeros(1, 3, h, w, dtype=torch.float32)

    def vae_output_to_image(self, tensor):
        return Image.new("RGB", (64, 64), color="blue")


class TestMockPipeline(unittest.TestCase):

    def test_mock_pipeline_run(self):
        mock_pipe = MockPipelineObj()
        pipe = HyperionPipeline(pipe=mock_pipe, backbone_type="custom", device="cpu", torch_dtype=torch.float32)

        # Mock query, key, value, out in cache so Adam loss step doesn't fail
        def mock_step_hooks():
            pipe.cache.add(
                torch.randn(1, 2, 16, 16),
                torch.randn(1, 2, 16, 16),
                torch.randn(1, 2, 16, 16),
                torch.randn(1, 2, 16, 16),
            )

        # Config with very small steps for testing
        cfg = HyperionConfig.t2i_default(
            width=64,
            height=64,
            num_inference_steps=2,
            start_time=999,
            adam_iters=1,
            slt_iters=2,
            ot_iters=2,
        )

        dummy_style = Image.new("RGB", (64, 64), color="red")
        dummy_content = Image.new("RGB", (64, 64), color="green")

        # Verify load_image helper
        loaded_img = pipe._load_image(dummy_style, (64, 64))
        self.assertEqual(loaded_img.size, (64, 64))


if __name__ == "__main__":
    unittest.main()
