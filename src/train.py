
import argparse
import yaml
import torch
from comet_ml import Experiment

import train, create_dataloader, build_model, save_model


# --------------------------------------------------
# Argument parsing
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


# --------------------------------------------------
# Main training function
# --------------------------------------------------
def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Init Comet
    experiment = Experiment(
        project_name="dog-breed-classification",
        workspace="bernandogunawan"
    )
    experiment.log_parameters(config)
    experiment.set_name(config["experiment"]["name"])
    experiment.add_tags(config["experiment"]["tags"])

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment.log_parameter("device", device)

    # Data
    train_dataloader, val_loader = get_dataloaders(config)

    # Model
    model = build_model(
        backbone=config["model"]["backbone"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"]
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Stage 1: Train classifier head (freeze backbone)
    # --------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["lr_head"]
    )

    for epoch in range(config["training"]["freeze_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        experiment.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        }, step=epoch)

    # --------------------------------------------------
    # Stage 2: Fine-tuning (unfreeze backbone)
    # --------------------------------------------------
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr_backbone"]
    )

    start_epoch = config["training"]["freeze_epochs"]

    for epoch in range(start_epoch, start_epoch + config["training"]["finetune_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        experiment.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        }, step=epoch)

    # Save final model
    torch.save(model.state_dict(), "final_model.pt")
    experiment.log_model("final_model", "final_model.pt")

    experiment.end()


if __name__ == "__main__":
    main()
