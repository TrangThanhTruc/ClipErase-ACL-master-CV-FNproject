import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
import copy
import os

from pathlib import Path
from string import Template

from vis_utils import generate_and_save_gradcam

logging.basicConfig(level=logging.INFO)

EPSILON = 1e-6


def calculate_loss(outputs, flags, settings, outputs_from_model_ori=None):
    """
    Computes multi-component loss for both forget and retain sets,
    including optional KL divergence from original model outputs.
    """
    batch_size: int = len(flags)
    logits_per_image, logits_per_text = (
        outputs["logits_per_image"],
        outputs["logits_per_text"],
    )
    device = logits_per_image.device

    # flags = True (1) la Forget Set theo forget_rule
    forget_set_mask = torch.tensor(flags, dtype=torch.long).to(device)

    # Nguoc lai la Retain Set (flags = False -> 0)
    retain_set_mask = torch.tensor(flags, dtype=torch.bool)
    retain_set_mask = ~retain_set_mask
    retain_set_mask = retain_set_mask.type(torch.long).to(device)

    targets = torch.arange(batch_size).to(torch.long).to(device)

    forget_set_image_loss = (
        forget_set_mask * F.cross_entropy(logits_per_image, targets, reduction="none")
    ).mean()
    forget_set_text_loss = (
        forget_set_mask * F.cross_entropy(logits_per_text, targets, reduction="none")
    ).mean()

    retain_set_image_loss = (
        retain_set_mask * F.cross_entropy(logits_per_image, targets, reduction="none")
    ).mean()
    retain_set_text_loss = (
        retain_set_mask * F.cross_entropy(logits_per_text, targets, reduction="none")
    ).mean()

    kl_div_image_loss = 0
    kl_div_text_loss = 0
    if outputs_from_model_ori:
        logits_per_image_from_model_ori, logits_per_text_from_model_ori = (
            outputs_from_model_ori["logits_per_image"],
            outputs_from_model_ori["logits_per_text"],
        )
        probs_per_image = F.softmax(logits_per_image, dim=-1)
        probs_per_text = F.softmax(logits_per_text, dim=-1)
        probs_per_image_from_model_ori = F.softmax(
            logits_per_image_from_model_ori, dim=-1
        )
        probs_per_text_from_model_ori = F.softmax(
            logits_per_text_from_model_ori, dim=-1
        )

        kl_div_image_loss = (
            retain_set_mask.unsqueeze(1)
            * F.kl_div(
                torch.log(probs_per_image + EPSILON),
                probs_per_image_from_model_ori,
                reduction="none",
            )
        ).mean()
        kl_div_text_loss = (
            retain_set_mask.unsqueeze(1)
            * F.kl_div(
                torch.log(probs_per_text + EPSILON),
                probs_per_text_from_model_ori,
                reduction="none",
            )
        ).mean()

    return {
        "forget_set_image_loss": forget_set_image_loss,
        "forget_set_text_loss": forget_set_text_loss,
        "retain_set_image_loss": retain_set_image_loss,
        "retain_set_text_loss": retain_set_text_loss,
        "kl_div_image_loss": kl_div_image_loss,
        "kl_div_text_loss": kl_div_text_loss,
    }


def calculate_acc(outputs, flags, settings):
    """
    Calculates retrieval error and accuracy metrics.
    """
    batch_size: int = len(flags)
    logits_per_image, logits_per_text = (
        outputs["logits_per_image"],
        outputs["logits_per_text"],
    )
    device = logits_per_image.device

    forget_set_mask = torch.tensor(flags, dtype=torch.long).to(device)

    retain_set_mask = torch.tensor(flags, dtype=torch.bool)
    retain_set_mask = ~retain_set_mask
    retain_set_mask = retain_set_mask.type(torch.long).to(device)

    targets = torch.arange(batch_size).to(torch.long).to(device)

    # Dem so luong du doan SAI (!= targets) cho Forget Set
    true_cnt_forget_image = (
        forget_set_mask
        * (torch.argmax(logits_per_image, dim=1) != targets).type(torch.int32)
    ).sum()
    false_cnt_forget_image = (
        forget_set_mask
        * (torch.argmax(logits_per_image, dim=1) == targets).type(torch.int32)
    ).sum()
    true_cnt_forget_text = (
        forget_set_mask
        * (torch.argmax(logits_per_text, dim=1) != targets).type(torch.int32)
    ).sum()
    false_cnt_forget_text = (
        forget_set_mask
        * (torch.argmax(logits_per_text, dim=1) == targets).type(torch.int32)
    ).sum()

    curr_cnt_forget_samples = forget_set_mask.sum().item()

    # Dem so luong du doan DUNG (== targets) cho Retain Set
    true_cnt_retain_image = (
        retain_set_mask
        * (torch.argmax(logits_per_image, dim=1) == targets).type(torch.int32)
    ).sum()
    false_cnt_retain_image = (
        retain_set_mask
        * (torch.argmax(logits_per_image, dim=1) != targets).type(torch.int32)
    ).sum()
    true_cnt_retain_text = (
        retain_set_mask
        * (torch.argmax(logits_per_text, dim=1) == targets).type(torch.int32)
    ).sum()
    false_cnt_retain_text = (
        retain_set_mask
        * (torch.argmax(logits_per_text, dim=1) != targets).type(torch.int32)
    ).sum()

    curr_cnt_retain_samples = retain_set_mask.sum().item()

    return {
        "true_cnt_forget_image": true_cnt_forget_image,
        "false_cnt_forget_image": false_cnt_forget_image,
        "true_cnt_forget_text": true_cnt_forget_text,
        "false_cnt_forget_text": false_cnt_forget_text,
        "curr_cnt_forget_samples": curr_cnt_forget_samples,
        "true_cnt_retain_image": true_cnt_retain_image,
        "false_cnt_retain_image": false_cnt_retain_image,
        "true_cnt_retain_text": true_cnt_retain_text,
        "false_cnt_retain_text": false_cnt_retain_text,
        "curr_cnt_retain_samples": curr_cnt_retain_samples,
    }


def train_one_epoch(
    epoch,
    model,
    train_loader,
    optimizer,
    tensorboard_writer,
    settings,
    checkpoint_helper,
    model_ori=None,
):
    mode: str = "train"
    device = torch.device(settings["misc"]["device"])
    epochs: int = settings[mode]["epochs"]

    model.train()
    pbar = tqdm(train_loader, total=len(train_loader), ncols=100)
    total_loss_epoch = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar, start=1):
        images, texts, flags = batch
        batch_size = len(flags)

        outputs = model(images, texts)

        if model_ori:
            model_ori.eval()
            with torch.no_grad():
                outputs_from_model_ori = model_ori(images, texts)
            loss_dict = calculate_loss(outputs, flags, settings, outputs_from_model_ori)
        else:
            loss_dict = calculate_loss(outputs, flags, settings)

        forget_set_image_loss = loss_dict["forget_set_image_loss"] * settings["loss"]["weights"].get("forget_set_image_loss", 1.0)
        forget_set_text_loss = loss_dict["forget_set_text_loss"] * settings["loss"]["weights"].get("forget_set_text_loss", 1.0)
        retain_set_image_loss = loss_dict["retain_set_image_loss"] * settings["loss"]["weights"].get("retain_set_image_loss", 1.0)
        retain_set_text_loss = loss_dict["retain_set_text_loss"] * settings["loss"]["weights"].get("retain_set_text_loss", 1.0)
        kl_div_image_loss = loss_dict.get("kl_div_image_loss", 0) * settings["loss"]["weights"].get("kl_div_image_loss", 1.0)
        kl_div_text_loss = loss_dict.get("kl_div_text_loss", 0) * settings["loss"]["weights"].get("kl_div_text_loss", 1.0)

        # Tong hop Loss: Tru vao Forget, Cong vao Retain va KL Div
        total_loss = (
            -forget_set_image_loss
            - forget_set_text_loss
            + retain_set_image_loss
            + retain_set_text_loss
            + kl_div_image_loss
            + kl_div_text_loss
        )
        total_loss_epoch += total_loss.item()

        pbar.set_description(
            f"Train: Epoch {epoch}/{epochs} [{batch_idx}/{len(train_loader)}], Loss: {total_loss.item():.4f}"
        )

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss_epoch = total_loss_epoch / len(train_loader)
    tensorboard_writer.add_scalar("Training loss", avg_loss_epoch, epoch)
    tensorboard_writer.add_scalar("Forget Set Image Loss", forget_set_image_loss, epoch)
    tensorboard_writer.add_scalar("Forget Set Text Loss", forget_set_text_loss, epoch)
    tensorboard_writer.add_scalar("Retain Set Image Loss", retain_set_image_loss, epoch)
    tensorboard_writer.add_scalar("Retain Set Text Loss", retain_set_text_loss, epoch)
    tensorboard_writer.add_scalar("KL Divergence Image Loss", kl_div_image_loss, epoch)
    tensorboard_writer.add_scalar("KL Divergence Text Loss", kl_div_text_loss, epoch)


def train(model, train_loader, optimizer, settings, checkpoint_helper, val_loader=None):
    mode: str = "train"
    epochs: int = settings[mode]["epochs"]

    writer = SummaryWriter(settings["train"]["tensorboard"])
    model_ori = copy.deepcopy(model)

    for epoch in range(1, epochs + 1):
        train_one_epoch(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            tensorboard_writer=writer,
            settings=settings,
            checkpoint_helper=checkpoint_helper,
            model_ori=model_ori,
        )
        if val_loader:
            generate_and_save_gradcam(
                copy.deepcopy(model),
                copy.deepcopy(val_loader.dataset),
                settings,
                epoch=epoch,
            )
            test(model, val_loader, settings, checkpoint_helper, epoch)

    writer.close()


def test(model, dataloader, settings, checkpoint_helper, epoch):
    mode: str = "eval"
    device = torch.device(settings["misc"]["device"])
    epochs: int = settings["train"]["epochs"]

    writer = SummaryWriter(settings["train"]["tensorboard"])

    checkpoint_template: Template = Template(
        os.path.join(settings["train"]["checkpoint"], "acc_${avg_acc}.pt")
    )

    model.eval()
    pbar = tqdm(dataloader, total=len(dataloader), ncols=100)
    total_loss_epoch = 0
    total_samples = 0
    total_cnt_forget_samples = 0
    total_cnt_retain_samples = 0
    
    total_img_ret_forget_error = 0
    total_text_ret_forget_error = 0
    total_img_ret_retain_true = 0
    total_text_ret_retain_true = 0

    with torch.no_grad():
        for batch in pbar:
            images, texts, flags = batch
            batch_size = len(flags)

            outputs = model(images, texts)

            loss_dict = calculate_loss(outputs, flags, settings)
            forget_set_image_loss = loss_dict["forget_set_image_loss"] * settings["loss"]["weights"].get("forget_set_image_loss", 1.0)
            forget_set_text_loss = loss_dict["forget_set_text_loss"] * settings["loss"]["weights"].get("forget_set_text_loss", 1.0)
            retain_set_image_loss = loss_dict["retain_set_image_loss"] * settings["loss"]["weights"].get("retain_set_image_loss", 1.0)
            retain_set_text_loss = loss_dict["retain_set_text_loss"] * settings["loss"]["weights"].get("retain_set_text_loss", 1.0)
            kl_div_image_loss = loss_dict.get("kl_div_image_loss", 0) * settings["loss"]["weights"].get("kl_div_image_loss", 1.0)
            kl_div_text_loss = loss_dict.get("kl_div_text_loss", 0) * settings["loss"]["weights"].get("kl_div_text_loss", 1.0)

            total_loss = (
                -forget_set_image_loss
                - forget_set_text_loss
                + retain_set_image_loss
                + retain_set_text_loss
                + kl_div_image_loss
                + kl_div_text_loss
            )

            acc_dict = calculate_acc(outputs, flags, settings)

            curr_cnt_forget_samples = acc_dict["curr_cnt_forget_samples"]
            curr_cnt_retain_samples = acc_dict["curr_cnt_retain_samples"]

            total_cnt_forget_samples += curr_cnt_forget_samples
            total_cnt_retain_samples += curr_cnt_retain_samples
            total_samples += batch_size * 4

            total_img_ret_forget_error += acc_dict["true_cnt_forget_image"]
            total_text_ret_forget_error += acc_dict["true_cnt_forget_text"]
            total_img_ret_retain_true += acc_dict["true_cnt_retain_image"]
            total_text_ret_retain_true += acc_dict["true_cnt_retain_text"]

            pbar.set_description(
                f"Test: Epoch {epoch}/{epochs}, Loss: {total_loss.item():.4f}"
            )

        # Tinh Error Rate cho tap Forget
        forget_image_error_rate = (total_img_ret_forget_error / total_cnt_forget_samples) if total_cnt_forget_samples > 0 else 0.0
        forget_text_error_rate = (total_text_ret_forget_error / total_cnt_forget_samples) if total_cnt_forget_samples > 0 else 0.0
        
        # Dao nguoc Error Rate thanh True Accuracy cho tap Forget de de quan sat
        total_image_ret_forget_set_acc = 1.0 - forget_image_error_rate
        total_text_ret_forget_set_acc = 1.0 - forget_text_error_rate

        total_image_ret_retain_set_acc = (total_img_ret_retain_true / total_cnt_retain_samples) if total_cnt_retain_samples > 0 else 0.0
        total_text_ret_retain_set_acc = (total_text_ret_retain_true / total_cnt_retain_samples) if total_cnt_retain_samples > 0 else 0.0

        logging.info(f"Total loss: {total_loss}")

        logging.info(
            f"Total Image Retrieval/0-shot ACCURACY in forget set: {total_image_ret_forget_set_acc * 100:.2f} % (Forget Rate: {forget_image_error_rate * 100:.2f}%)"
        )
        logging.info(
            f"Total Image Retrieval/0-shot ACCURACY in retain set: {total_image_ret_retain_set_acc * 100:.2f} %"
        )
        logging.info(
            f"Total Text Retrieval ACCURACY in forget set: {total_text_ret_forget_set_acc * 100:.2f} % (Forget Rate: {forget_text_error_rate * 100:.2f}%)"
        )
        logging.info(
            f"Total Text Retrieval ACCURACY in retain set: {total_text_ret_retain_set_acc * 100:.2f} %"
        )

        writer.add_scalar("Test Loss", total_loss, epoch)
        writer.add_scalar("Test Image Retrieval/0-shot Accuracy per Forget Set", total_image_ret_forget_set_acc, epoch)
        writer.add_scalar("Test Image Retrieval/0-shot Accuracy per Retain Set", total_image_ret_retain_set_acc, epoch)
        writer.add_scalar("Test Text Retrieval Accuracy per Forget Set", total_text_ret_forget_set_acc, epoch)
        writer.add_scalar("Test Text Retrieval Accuracy per Retain Set", total_text_ret_retain_set_acc, epoch)

        avg_acc = (
            total_image_ret_forget_set_acc
            + total_image_ret_retain_set_acc
            + total_text_ret_forget_set_acc
            + total_text_ret_retain_set_acc
        ) / 4.0
        
        safe_avg_acc = float(avg_acc if isinstance(avg_acc, float) else avg_acc.item())
        checkpoint_helper.save_model_when_max_recall(
            total_loss,
            model,
            checkpoint_template.substitute(avg_acc=safe_avg_acc),
        )