
import argparse
import yaml
from comet_ml import Experiment
import torch

import utils, data_setup, engine, model_builder


# --------------------------------------------------
# Argument parsing
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--freeze_epochs", type=int)
    parser.add_argument("--unfreeze_epochs", type=int)
    parser.add_argument("--freeze_lr", type=float)
    parser.add_argument("--unfreeze_lr", type=float)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


# --------------------------------------------------
# Main training function
# --------------------------------------------------
def main():
    args = parse_args()

    config = {
        "data": {},
        "model": {},
        "training": {},
        "experiment": {}
    }
    # Load config
    with open(args.config) as f:
        config.update(yaml.safe_load(f))


    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.freeze_epochs is not None:
        config["training"]["freeze_epochs"] = args.freeze_epochs
    if args.unfreeze_epochs is not None:
        config["training"]["unfreeze_epochs"] = args.unfreeze_epochs
    if args.freeze_lr is not None:
        config["training"]["freeze_lr"] = args.freeze_lr
    if args.unfreeze_lr is not None:
        config["training"]["unfreeze_lr"] = args.unfreeze_lr

    # Init Comet
    experiment = Experiment(
        "5Cd7hajJ3PuFPv85lVNcmB2lm",
        project_name="dog-breed-classification",
        workspace="bernandogunawan"
    )
    experiment.log_parameters(config)
    experiment.set_name(config["experiment"]["name"])
    experiment.add_tags(config["experiment"]["tags"])

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment.log_parameter("device", device)

    # Model
    model,weight = model_builder.build_model(
        model_choice=config["model"]["name"],
        num_classes=config["model"]["num_classes"]
    )
    model = model.to(device)

    # preprocess
    train_dataloader,test_dataloader,class_name = data_setup.create_dataloader(
        path=args.path,
        transform=weight,
        image_size=config["data"]["img_size"],
        batch_size=config["data"]["batch_size"]
    )
    loss_func = torch.nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Stage 1: Train classifier head (freeze backbone)
    # --------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"]
    )

    for epoch in range(config["training"]["freeze_epochs"]):
        train_loss, train_acc = engine.train_step(
            model, train_dataloader, optimizer, loss_func)
        test_loss, test_acc = engine.evaluate(
            model, test_dataloader, loss_func)

        experiment.log_metrics({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": test_loss,
            "val_acc": test_acc
        }, step=epoch)

    # --------------------------------------------------
    # Stage 2: Fine-tuning (unfreeze backbone)
    # --------------------------------------------------
    # for param in model.parameters():
    #     param.requires_grad = True

    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=config["training"]["lr_backbone"]
    # )

    # start_epoch = config["training"]["freeze_epochs"]

    # for epoch in range(start_epoch, start_epoch + config["training"]["finetune_epochs"]):
    #     train_loss = train_one_epoch(
    #         model, train_loader, criterion, optimizer, device
    #     )
    #     val_loss, val_acc = validate(
    #         model, val_loader, criterion, device
    #     )

    #     experiment.log_metrics({
    #         "train_loss": train_loss,
    #         "val_loss": val_loss,
    #         "val_acc": val_acc
    #     }, step=epoch)

    # Save final model
    # torch.save(model.state_dict(), "final_model.pt")
    # experiment.log_model("final_model", "final_model.pt")

    experiment.end()


if __name__ == "__main__":
    main()
