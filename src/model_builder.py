
import torchvision.models as models
import torch.nn as nn

def build_model(model_choice, num_classes, pretrained=True):
    if model_choice == "efficientnet_b2":
        weight = models.EfficientNet_B2_Weights.DEFAULT
        model = models.efficientnet_b2(weights=weight)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        

        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_choice == "resnet50":
        weight = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weight)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Unsupported model choice")

    return model, weight.transforms()
