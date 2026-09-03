import importlib.util
import tempfile
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

    def test_warning_checkpoint_budget_allows_exactly_twenty(self):
        events = []
        budget = FORENSIC.CheckpointBudget(
            max_warning_checkpoints=20,
            max_total_checkpoints=40,
            event_sink=events.append,
        )
        saved = 0
        for index in range(100):
            path = f"warning_{index}.pt"
            if budget.can_save(
                checkpoint_kind="warning",
                path=path,
                context={"epoch": 1, "batch": index, "split": "train"},
            ):
                budget.record_saved(checkpoint_kind="warning", path=path)
                saved += 1
        self.assertEqual(saved, 20)
        self.assertEqual(budget.warning_checkpoints_saved, 20)
        self.assertEqual(budget.ordinary_checkpoints_saved, 20)
        self.assertEqual(
            [event["event_type"] for event in events],
            ["checkpoint_budget_exhausted"],
        )

    def test_global_checkpoint_budget_caps_ordinary_files(self):
        events = []
        budget = FORENSIC.CheckpointBudget(
            max_warning_checkpoints=100,
            max_total_checkpoints=40,
            event_sink=events.append,
        )
        saved = 0
        for index in range(100):
            path = f"regular_{index}.pt"
            if budget.can_save(
                checkpoint_kind="regular",
                path=path,
                context={"epoch": index + 1, "batch": -1, "split": "epoch"},
            ):
                budget.record_saved(checkpoint_kind="regular", path=path)
                saved += 1
        self.assertEqual(saved, 40)
        self.assertTrue(budget.total_budget_exhausted)
        self.assertEqual(
            [event["event_type"] for event in events],
            ["checkpoint_budget_exhausted"],
        )

    def test_failure_checkpoints_ignore_ordinary_budget(self):
        budget = FORENSIC.CheckpointBudget(
            max_warning_checkpoints=0,
            max_total_checkpoints=0,
        )
        self.assertFalse(
            budget.can_save(
                checkpoint_kind="warning",
                path="warning.pt",
                context={"epoch": 1, "batch": 1, "split": "train"},
            )
        )
        with tempfile.TemporaryDirectory(prefix="forensic_failure_budget_") as directory:
            directory_path = Path(directory)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=100,
                eta_min=1e-6,
            )
            observer = FORENSIC.ForensicObserver()
            observer.begin_forward(
                split="train",
                epoch=1,
                batch_index=1,
                global_step=1,
                sample_ids=["synthetic"],
            )
            failure = FORENSIC.ForensicFailure(
                phase="gradient",
                stage="gradient",
                tensor_name="weight",
                stats=FORENSIC.tensor_stats(torch.tensor([float("nan")])),
            )
            FORENSIC.failure_report_and_state(
                output_dir=directory_path,
                failure=failure,
                observer=observer,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config={"seed": 42},
                epoch=1,
                batch_index=1,
                global_step=1,
                sample_ids=["synthetic"],
                learning_rate=1e-3,
                gradient_info=None,
                best_validation_dice=float("-inf"),
                train_generator=torch.Generator().manual_seed(42),
                validation_generator=torch.Generator().manual_seed(43),
                recent_history=[],
            )
            self.assertTrue(
                (directory_path / "checkpoints" / "pre_failure_step.pt").is_file()
            )
            self.assertTrue(
                (directory_path / "checkpoints" / "failure_state.pt").is_file()
            )

    def test_gradient_parameter_and_optimizer_summaries_are_scalar_and_finite(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        gradients = FORENSIC.gradient_summary(model)
        parameters_before = FORENSIC.parameter_summary(model)
        optimizer_before = FORENSIC.optimizer_state_summary(model, optimizer)
        optimizer.step()
        parameters_after = FORENSIC.parameter_summary(model)
        optimizer_after = FORENSIC.optimizer_state_summary(model, optimizer)
        self.assertGreater(gradients["max_abs"], 0.0)
        self.assertIsNotNone(gradients["min"])
        self.assertIsNotNone(gradients["parameter_name_with_max_abs"])
        self.assertFalse(gradients["has_nan"] or gradients["has_inf"])
        self.assertFalse(parameters_before["has_nan"] or parameters_before["has_inf"])
        self.assertFalse(parameters_after["has_nan"] or parameters_after["has_inf"])
        self.assertFalse(optimizer_before["has_state"])
        self.assertTrue(optimizer_after["has_state"])
        self.assertIsNotNone(optimizer_after["max_exp_avg_abs"])
        self.assertIsNotNone(optimizer_after["max_exp_avg_sq_abs"])

    def test_nonfinite_gradient_optimizer_state_and_parameter_are_classified(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.weight.grad = torch.full_like(model.weight, float("nan"))
        gradient_failure = FORENSIC.gradient_summary(model)["first_nonfinite"]
        self.assertEqual(gradient_failure["parameter_name"], "weight")

        optimizer.zero_grad(set_to_none=False)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.state[model.weight]["exp_avg"].fill_(float("nan"))
        optimizer_failure = FORENSIC.optimizer_state_summary(model, optimizer)["first_nonfinite"]
        self.assertEqual(optimizer_failure["parameter_name"], "weight")
        self.assertEqual(optimizer_failure["state_name"], "exp_avg")

        model.bias.data.fill_(float("nan"))
        parameter_failure = FORENSIC.parameter_summary(model)["first_nonfinite"]
        self.assertEqual(parameter_failure["parameter_name"], "bias")

    def test_failure_classification_distinguishes_loss_from_forward(self):
        loss_failure = FORENSIC.ForensicFailure(
            phase="loss",
            stage="total_loss",
            tensor_name="global/total_loss",
            stats=FORENSIC.tensor_stats(torch.tensor([float("nan")])),
        )
        forward_failure = FORENSIC.ForensicFailure(
            phase="forward",
            stage="segmentation_logits",
            tensor_name="global/segmentation_logits",
            stats=FORENSIC.tensor_stats(torch.tensor([float("nan")])),
        )
        self.assertEqual(
            FORENSIC.classify_failure(loss_failure),
            "loss",
        )
        self.assertEqual(
            FORENSIC.classify_failure(forward_failure),
            "forward_activation",
        )

    def test_optimization_monitoring_does_not_change_updates(self):
        torch.manual_seed(42)
        monitored = torch.nn.Linear(2, 1)
        reference = torch.nn.Linear(2, 1)
        reference.load_state_dict(monitored.state_dict())
        monitored_optimizer = torch.optim.AdamW(monitored.parameters(), lr=1e-3)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        inputs = torch.ones(1, 2)
        monitored_loss = monitored(inputs).sum()
        monitored_loss.backward()
        FORENSIC.gradient_summary(monitored)
        FORENSIC.parameter_summary(monitored)
        FORENSIC.optimizer_state_summary(monitored, monitored_optimizer)
        monitored_optimizer.step()
        FORENSIC.parameter_summary(monitored)
        FORENSIC.optimizer_state_summary(monitored, monitored_optimizer)

        reference_optimizer.zero_grad(set_to_none=True)
        reference(inputs).sum().backward()
        reference_optimizer.step()
        self.assertEqual(
            FORENSIC.compare_nested_tensors(
                monitored.state_dict(), reference.state_dict()
            ),
            0.0,
        )
        self.assertEqual(
            FORENSIC.compare_nested_tensors(
                monitored_optimizer.state_dict(), reference_optimizer.state_dict()
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
