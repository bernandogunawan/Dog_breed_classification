
import torchvision.models as models
import torch.nn as nn

def build_model(model_choice, num_classes, pretrained=True):
    if model_choice == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    elif model_choice == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError("Unsupported model choice")

    return model
