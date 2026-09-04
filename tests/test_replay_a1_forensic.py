import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "replay_a1_forensic.py"
SPEC = importlib.util.spec_from_file_location("replay_a1_forensic", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class SyntheticDataset:
    image_size = (224, 224)

    def __init__(self):
        self.entries = [{"id": "synthetic-0", "image": "synthetic-0.jpg"}]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return {
            "image": torch.zeros(3, 224, 224),
            "mask": torch.zeros(1, 224, 224),
            "id": self.entries[index]["id"],
        }


class ReplayA1ForensicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = REPLAY.load_config(PROJECT_ROOT / "configs/vil_bottleneck_a1.json")

    def test_checkpoint_loading_and_deterministic_selection(self):
        torch.manual_seed(42)
        model = REPLAY.build_model(self.config)
        with tempfile.TemporaryDirectory(prefix="replay_a1_checkpoint_") as directory:
            checkpoint = Path(directory) / "synthetic.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": self.config,
                    "epoch": 15,
                    "global_step": 123,
                    "seed": 42,
                },
                checkpoint,
            )
            config, payload, loaded = REPLAY.load_replay_model(
                checkpoint,
                PROJECT_ROOT / "configs/vil_bottleneck_a1.json",
                torch.device("cpu"),
            )
            self.assertEqual(payload["epoch"], 15)
            self.assertEqual(payload["global_step"], 123)
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(config["model"]["name"], self.config["model"]["name"])
            self.assertEqual(
                REPLAY.select_sample_indices(SyntheticDataset(), [0]),
                [0],
            )

    def test_statistics_json_and_inference_equivalence(self):
        torch.manual_seed(42)
        reference = REPLAY.build_model(self.config).eval()
        instrumented_model = REPLAY.build_model(self.config).eval()
        instrumented_model.load_state_dict(reference.state_dict())
        dataset = SyntheticDataset()
        image = dataset[0]["image"].unsqueeze(0)
        with torch.no_grad():
            reference_output = reference(image)

        observer = REPLAY.FORENSIC.ForensicObserver()
        instrumentation = REPLAY.FORENSIC.A1Instrumentation(instrumented_model, observer)
        instrumentation.install()
        instrumentation.activate()
        try:
            replay = REPLAY.collect_sample_replay(
                dataset=dataset,
                index=0,
                model=instrumented_model,
                device=torch.device("cpu"),
                observer=observer,
                epoch=15,
                global_step=123,
                split="test",
            )
            with torch.no_grad():
                instrumented_output = instrumented_model(image)
        finally:
            instrumentation.deactivate()
            instrumentation.close()

        self.assertTrue(torch.equal(reference_output, instrumented_output))
        self.assertEqual(replay["id"], "synthetic-0")
        self.assertIn("global/cnn_bottleneck_input", observer.max_observed)
        self.assertTrue(replay["statistics"]["forward"])
        self.assertTrue(replay["statistics"]["reverse"])
        for direction in ("forward", "reverse"):
            block = replay["statistics"][direction][sorted(replay["statistics"][direction])[0]]
            for measurement in REPLAY.OBSERVED_VIL_MEASUREMENTS:
                self.assertIn(measurement, block)
        self.assertIn(
            "forward_final_abs_max_over_cnn_bottleneck_abs_max",
            replay["amplification_ratios"],
        )
        with tempfile.TemporaryDirectory(prefix="replay_a1_json_") as directory:
            output = Path(directory) / "replay.json"
            REPLAY.write_json(output, replay)
            decoded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(decoded["id"], "synthetic-0")
            self.assertIn("statistics", decoded)


if __name__ == "__main__":
    unittest.main()
