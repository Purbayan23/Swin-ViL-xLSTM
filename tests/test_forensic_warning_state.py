import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_a1_forensic.py"
SPEC = importlib.util.spec_from_file_location("train_a1_forensic", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
FORENSIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FORENSIC)


def observe(
    observer,
    *,
    split: str,
    epoch: int,
    batch: int,
    value: float,
    name: str = "test_metric",
) -> bool:
    observer.begin_forward(
        split=split,
        epoch=epoch,
        batch_index=batch,
        global_step=epoch * 100 + batch,
        sample_ids=["synthetic"],
    )
    observer.record(
        "vil/forward_block_0",
        name,
        torch.tensor([value]),
    )
    _, _, crossed = observer.finish_forward()
    if crossed:
        observer.record_warning_checkpoint_saved("synthetic_warning.pt")
    observer.clear()
    return crossed


class ForensicWarningStateTest(unittest.TestCase):
    def test_100_consecutive_above_threshold_batches_create_one_checkpoint(self):
        observer = FORENSIC.ForensicObserver()
        checkpoints = sum(
            observe(
                observer,
                split="train",
                epoch=(index // 10) + 1,
                batch=index % 10,
                value=101.0,
            )
            for index in range(100)
        )
        self.assertEqual(checkpoints, 1)

    def test_epoch_boundary_preserves_continuous_episode(self):
        observer = FORENSIC.ForensicObserver()
        checkpoints = 0
        for epoch in (1, 2):
            for batch in range(10):
                checkpoints += observe(
                    observer,
                    split="train",
                    epoch=epoch,
                    batch=batch,
                    value=101.0,
                )
        self.assertEqual(checkpoints, 1)

    def test_train_and_validation_are_independent_streams(self):
        observer = FORENSIC.ForensicObserver()
        train_checkpoints = sum(
            observe(
                observer,
                split="train",
                epoch=1,
                batch=batch,
                value=101.0,
            )
            for batch in range(10)
        )
        validation_checkpoints = sum(
            observe(
                observer,
                split="validation",
                epoch=1,
                batch=batch,
                value=101.0,
            )
            for batch in range(10)
        )
        self.assertEqual(train_checkpoints, 1)
        self.assertEqual(validation_checkpoints, 1)

    def test_false_true_true_false_true_creates_two_checkpoints(self):
        observer = FORENSIC.ForensicObserver()
        values = (50.0, 101.0, 110.0, 50.0, 101.0)
        checkpoints = sum(
            observe(
                observer,
                split="train",
                epoch=1,
                batch=batch,
                value=value,
            )
            for batch, value in enumerate(values)
        )
        self.assertEqual(checkpoints, 2)

    def test_negative_log_decay_and_lifecycle_fields(self):
        observer = FORENSIC.ForensicObserver()
        values = (-49.0, -51.0, -55.0, -49.0, -51.0)
        checkpoints = sum(
            observe(
                observer,
                split="train",
                epoch=1,
                batch=batch,
                value=value,
                name="max_log_decay",
            )
            for batch, value in enumerate(values)
        )
        self.assertEqual(checkpoints, 2)
        event_types = {event["event_type"] for event in observer.warning_lifecycle_events}
        self.assertTrue(
            {
                "warning_state_created",
                "warning_condition_true",
                "warning_episode_started",
                "warning_state_reset",
                "warning_checkpoint_saved",
            }.issubset(event_types)
        )
        required_fields = {
            "epoch",
            "batch",
            "split",
            "scope",
            "metric",
            "threshold",
            "previous_state",
            "current_state",
            "episode_id",
        }
        self.assertTrue(
            all(required_fields.issubset(event) for event in observer.warning_lifecycle_events)
        )


if __name__ == "__main__":
    unittest.main()
