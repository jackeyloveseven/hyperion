import unittest
from PIL import Image
import torch
import torch.nn as nn
from hyperion.pipeline.config import HyperionConfig
from hyperion.pipeline.hyperion_pipeline import HyperionPipeline


class MockDiT(nn.Module):
    def __init__(self, cache):
        super().__init__()
        self.conv = nn.Linear(128, 128)
        self.cache = cache

    def forward(
        self,
        hidden_states,
        timestep=None,
        guidance=None,
        encoder_hidden_states=None,
        txt_ids=None,
        img_ids=None,
    ):
        out = self.conv(hidden_states)
        b, l, c = hidden_states.shape
        q = hidden_states.reshape(b, 2, l, c // 2)
        so = out.reshape(b, 2, l, c // 2)
        self.cache.add(q, q, q, so)
        return out


class MockVAE(nn.Module):
    def encode(self, pix):
        b, c, h, w = pix.shape
        return torch.randn(b, 128, h // 16, w // 16, device=pix.device, dtype=pix.dtype)

    def decode(self, lat):
        b, c, h, w = lat.shape
        return torch.randn(b, 3, h * 16, w * 16, device=lat.device, dtype=lat.dtype)


class MockPipelineObj:
    def __init__(self):
        self.dit = None
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

    def setUp(self):
        self.mock_pipe = MockPipelineObj()
        self.pipe = HyperionPipeline(
            pipe=self.mock_pipe,
            backbone_type="custom",
            device="cpu",
            torch_dtype=torch.float32,
        )
        self.mock_pipe.dit = MockDiT(self.pipe.cache)
        self.dummy_style = Image.new("RGB", (64, 64), color="red")
        self.dummy_content = Image.new("RGB", (64, 64), color="green")

    def test_mock_pipeline_t2i_hybrid(self):
        cfg = HyperionConfig.t2i_default(
            width=64,
            height=64,
            num_inference_steps=2,
            start_time=999,
            adam_iters=1,
            slt_iters=2,
            ot_iters=2,
            mode="hybrid",
        )
        img = self.pipe.generate_t2i("A test prompt", self.dummy_style, config=cfg)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (64, 64))

    def test_mock_pipeline_i2i_hybrid(self):
        cfg = HyperionConfig.i2i_default(
            width=64,
            height=64,
            num_inference_steps=2,
            start_time=999,
            adam_iters=1,
            slt_iters=2,
            ot_iters=2,
            mode="hybrid",
        )
        img = self.pipe.generate_i2i(
            content_image=self.dummy_content,
            style_image=self.dummy_style,
            prompt="masterpiece",
            config=cfg,
        )
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (64, 64))

    def test_mock_pipeline_injection_and_vshuffle(self):
        for mode in ("inject", "vshuffle"):
            cfg = HyperionConfig.t2i_default(
                width=64,
                height=64,
                num_inference_steps=2,
                start_time=999,
                mode=mode,
            )
            img = self.pipe.generate_t2i("A test prompt", self.dummy_style, config=cfg)
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.size, (64, 64))

    def test_mock_pipeline_slt_and_adam_only(self):
        for mode in ("slt_only", "adam_only"):
            cfg = HyperionConfig.t2i_default(
                width=64,
                height=64,
                num_inference_steps=2,
                start_time=999,
                adam_iters=1,
                slt_iters=2,
                mode=mode,
            )
            img = self.pipe.generate_t2i("A test prompt", self.dummy_style, config=cfg)
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.size, (64, 64))


if __name__ == "__main__":
    unittest.main()
