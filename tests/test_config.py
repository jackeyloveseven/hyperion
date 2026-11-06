import tempfile
import unittest
from pathlib import Path
from hyperion.pipeline.config import HyperionConfig


class TestConfig(unittest.TestCase):

    def test_config_defaults_match_tpami(self):
        t2i = HyperionConfig.t2i_default()
        self.assertEqual(t2i.start_time, 850)
        self.assertEqual(t2i.adain_warm_alpha, 0.50)
        self.assertEqual(t2i.adam_iters, 2)
        self.assertEqual(t2i.slt_alpha, 0.10)
        self.assertEqual(t2i.freq_hi_boost, 2.0)
        self.assertEqual(t2i.guidance_scale, 3.5)

        i2i = HyperionConfig.i2i_default()
        self.assertEqual(i2i.start_time, 870)
        self.assertEqual(i2i.adain_warm_alpha, 0.30)
        self.assertEqual(i2i.adam_iters, 2)

    def test_config_serialization(self):
        cfg = HyperionConfig.t2i_default(seed=1234, num_inference_steps=30)
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "config.json"
            cfg.save_json(json_path)

            loaded = HyperionConfig.from_json(json_path)
            self.assertEqual(loaded.seed, 1234)
            self.assertEqual(loaded.num_inference_steps, 30)
            self.assertEqual(loaded.start_time, 850)
    def test_json_config_files(self):
        repo_root = Path(__file__).resolve().parent.parent
        t2i_path = repo_root / "configs" / "tpami_t2i_default.json"
        i2i_path = repo_root / "configs" / "tpami_i2i_default.json"

        self.assertTrue(t2i_path.exists())
        self.assertTrue(i2i_path.exists())

        t2i_cfg = HyperionConfig.from_json(t2i_path)
        self.assertEqual(t2i_cfg.start_time, 850)
        self.assertEqual(t2i_cfg.inject_beta, 0.50)

        i2i_cfg = HyperionConfig.from_json(i2i_path)
        self.assertEqual(i2i_cfg.start_time, 870)
        self.assertEqual(i2i_cfg.inject_beta, 0.50)


if __name__ == "__main__":
    unittest.main()
