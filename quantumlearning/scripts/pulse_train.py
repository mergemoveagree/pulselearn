"""Script to train a single pulse model.

.. codeauthor: Jaden Nola <a733p959@wichita.edu>
"""
from datetime import datetime
import logging
import math
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import lib.log.log as pulse_log

from lib.pytorch.datasets import transformations
from lib.pytorch.modules.dynamics_simulator_modules import DynamicsSimulatorNet
from utils.pulse_utils import wide_sigmoid
from utils import script_utils

log = logging.getLogger(pulse_log.ROOT_LOGGER_NAME + "." + __name__)


def main(params: dict[str, Any]) -> int:
    """Run a pulse model with options specified in params."""
    timestamp = datetime.now().strftime("%Y.%m.%d_%H%M%S")

    # Simulation options
    gate = transformations.get_gate(params["classifier"], params["transformation"])
    ql_backend: str = params["ql backend"]
    wrapper = script_utils.parse_ql_backend(ql_backend)(
        params["qubit frequencies"],
        params["rabi rate strengths"],
        params["time sampling rate"],
        params["qubit couplings"],
    )
    should_multiprocess: bool = params.get("multiprocessing", False)

    # Model options
    epochs: int = params["epochs"]
    cyclic_lr: bool = params.get("cyclic learning rate", False)
    min_lr: float = params["min learning rate"]
    max_lr: float | None = params.get("max learning rate", None)
    pulse_array: list[int] = params["pulse array"]
    target_qubit_index: int = params["target qubit"]
    extra_args: dict[str, Any] = params.get("extra args", {})
    initial_parameters: dict[str, list[list[float]]] | None = params.get(
        "initial parameters", None
    )

    # Dataset options
    training_points: int | None = params.get("training points", None)
    validation_points: int | None = params.get("validation points", None)
    train_data_path: str | None = params.get("training data path", None)
    valid_data_path: int | None = params.get("validation data path", None)
    batch_size: int = params["batch size"]

    script_utils.log_marker("Beginning model setup")

    if should_multiprocess:
        log.info("Using multiprocessing for backward pass")
    else:
        log.info("Running process on single core")

    if initial_parameters:
        log.info("Found initial parameters in testfile")
    else:
        log.info("No initial parameters found in testfile")

    # ---------- Create model name ----------
    if "model name" in params:
        model_name = params["model name"]
    else:  # No user-specified model name in testfile, generating default one
        model_name = f"{gate.name}_{timestamp}_init{int(bool(initial_parameters))}"
        if "model name tag" in params:
            model_name += f"_{params['model_name_tag']}"

    # ---------- Create run path directories ----------
    if pulse_log.run_root is None:
        raise ValueError("Run root was not properly set up")
    run_path = pulse_log.run_root / model_name
    run_path.mkdir()
    (run_path / "models").mkdir()
    (run_path / "tensorboard").mkdir()
    log.info("Using run path: %s", run_path)

    # Model tracking for saving models
    best_validation_loss = (
        1000000.0  # Impossible number so that the first model will overwrite this
    )
    best_model_log: list[str] = []

    # ---------- Get datasets for training and validation ----------
    if training_points is None and train_data_path is None:
        raise ValueError("No training points or dataset was provided")

    if train_data_path:
        # TODO Implement custom datasets
        raise ValueError("Custom training datasets is not yet supported")
    else:
        log.info("Generating training dataset...")
        train_dataset = gate.get_dataset(
            wrapper.num_qubits, training_points, target_qubit_index, **extra_args
        )
        log.info("Loading training dataset...")
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        log.info("Training dataset loaded")

    if validation_points is None and valid_data_path is None:
        raise ValueError("No validation points or dataset was provided")

    if valid_data_path:
        # TODO Implement custom datasets
        raise ValueError("Custom validation datasets is not yet supported")
    else:
        log.info("Generating validation dataset...")
        valid_dataset = gate.get_dataset(
            wrapper.num_qubits, validation_points, target_qubit_index, **extra_args
        )
        log.info("Loading validation dataset...")
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        log.info("Validation dataset loaded")

    log.info(
        "Running %s epochs with %s batches (batch size %s)",
        epochs,
        len(train_loader),
        batch_size,
    )
    log.info("Validating with %s validation points", len(valid_loader))

    # ---------- Set up Tensorboard logging ----------
    writer = SummaryWriter(str((run_path / "tensorboard").resolve()))

    # Prefer to log 10 times per epoch
    batch_logging_frequency = math.ceil(len(train_loader) / 10)
    if batch_logging_frequency == 1:
        log.info("Logging every batch")
    else:
        log.info("Logging every %s batches", batch_logging_frequency)

    # ---------- Create pulse model ----------
    log.info("Creating pulse model...")
    model = DynamicsSimulatorNet(
        wrapper, pulse_array, should_multiprocess, initial_parameters
    )
    log.info("Model successfully created")
    optimizer = torch.optim.SGD(model.parameters(), lr=min_lr, momentum=0.9)
    scheduler = None
    if cyclic_lr:
        if max_lr is None:
            raise ValueError(
                "'cyclic learning rate' is true, but no maximum learning rate was "
                "provided"
            )
        log.info("Using cyclic learning rate from %s to %s", min_lr, max_lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=min_lr, max_lr=max_lr
        )
    loss_function = torch.nn.MSELoss()

    script_utils.log_marker("Training pulse model")
    log.info("Initial model parameters:")
    model.log_params_to_stream()
    model.log_params_to_tensorboard(writer, 0)

    for epoch in range(epochs):
        log.info(
            "--------------------------------------------------------------------------------"
        )
        log.info("EPOCH %s", epoch + 1)
        model.train()
        running_loss = 0.0
        last_loss = 0.0
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            output = model(data, target)
            log.debug("Model output: %s", output.tolist())
            loss = loss_function(output, torch.ones(output.shape))
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running_loss += loss.item()
            points_index = epoch * len(train_loader) + i + 1
            model.log_params_to_tensorboard(writer, points_index)
            batch_print = (" " if i + 1 < 10 else "") + str(i + 1)
            if i % batch_logging_frequency == batch_logging_frequency - 1:
                last_loss = running_loss / batch_logging_frequency
                log.info("    batch %s  loss: %s", batch_print, last_loss)
                writer.add_scalar(  # pyright: ignore
                    "Loss/train", last_loss, points_index
                )
                running_loss = 0.0
            else:
                log.debug(
                    "    batch %s of epoch %s: loss %s",
                    batch_print,
                    epoch + 1,
                    loss.item(),
                )

        log.info("Model parameters after training in epoch %s", epoch + 1)
        model.log_params_to_stream()

        model.eval()

        # Running against training dataset
        running_validation_lost = 0.0
        for i, (vinputs, vlabels) in enumerate(valid_loader):
            voutputs = model(vinputs, vlabels)
            vloss = loss_function(voutputs, torch.ones(voutputs.shape))
            running_validation_lost += vloss.item()

        avg_vloss = running_validation_lost / len(valid_loader)
        log.info("LOSS train %s valid %s", last_loss, avg_vloss)
        writer.add_scalar("Loss/validation", avg_vloss, epoch + 1)  # pyright: ignore
        writer.flush()

        # If the average validation loss is better, we'll save the model
        if avg_vloss < best_validation_loss:
            best_validation_loss = avg_vloss
            best_model_log.clear()
            for name, param in model.named_parameters():
                report = param.detach().numpy()
                best_model_log.append(f"    {name} {report}")
                if "moduli" in name:
                    best_model_log.append(
                        f"        Effective moduli: {wide_sigmoid(report)}"
                    )
            model_path = str(run_path / "models" / f"pulse_model_{epoch}")
            torch.save(model.state_dict(), model_path)  # pyright: ignore
    writer.close()

    log.info("Best model parameters:")
    for line in best_model_log:
        log.info(line)

    return 0


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)  # pyright: ignore
    pulse_log.init_logging()
    params = script_utils.get_params()
    raise SystemExit(main(params))
