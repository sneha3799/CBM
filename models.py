from template_model import MLP, End2EndModel, BcosResNet18_Custom, ResNet18_Custom
import torch.nn as nn

# Independent & Sequential Model
def ModelXtoC(pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim, three_class, n_class_attr=2):
    return ResNet18_Custom(n_attributes=n_attributes, pretrained=pretrained, freeze=freeze, n_class_attr=n_class_attr)

# Independent Model
def ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part
    if n_class_attr == 3:
        model = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
    return model

# Sequential Model
def ModelXtoChat_ChatToY(n_class_attr, n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part (same as Independent model)
    return ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim)

# Joint Model
# def ModelXtoCtoY(n_class_attr, pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim,
#                  use_relu, use_sigmoid):
    
#     model1 = ResNet18_Custom(
#         n_attributes=n_attributes,
#         pretrained=pretrained,
#         freeze=freeze,
#         # expand_dim=expand_dim
#     )

#     if n_class_attr == 3:
#         model2 = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
#     else:
#         model2 = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)

#     return End2EndModel(model1, model2, use_relu, use_sigmoid, n_class_attr)

def ModelXtoCtoY(n_class_attr, pretrained, freeze, num_classes, n_attributes, use_relu, use_sigmoid):
    model1 = BcosResNet18_Custom(
        pretrained=pretrained,
        freeze=freeze,
        num_classes=num_classes,
        n_attributes=n_attributes,
        connect_CY=False,
        three_class=(n_class_attr == 3),
    )
    # Secondary MLP for C→Y mapping
    if n_class_attr == 3:
        model2 = nn.Sequential(
            nn.Linear(n_attributes * n_class_attr, num_classes),
            nn.ReLU() if use_relu else nn.Identity(),
        )
    else:
        model2 = nn.Sequential(
            nn.Linear(n_attributes, num_classes),
            nn.ReLU() if use_relu else nn.Identity(),
        )
    return End2EndModel(model1, model2, use_relu, use_sigmoid, n_class_attr)


# Standard Model (X → Y)
def ModelXtoY(pretrained, freeze, num_classes, use_aux):
    return ResNet18_Custom(pretrained=pretrained, freeze=freeze, num_classes=num_classes)


# Multitask Model (X → C, Y)
# def ModelXtoCY(pretrained, freeze, num_classes, use_aux,
#                n_attributes, three_class, connect_CY):
#     return ResNet18_Custom(pretrained=pretrained, freeze=freeze,
#                            num_classes=num_classes, n_attributes=n_attributes,
#                            three_class=three_class, connect_CY=connect_CY)
def ModelXtoCY(pretrained, freeze, num_classes, use_aux, n_attributes, three_class, connect_CY):
    return BcosResNet18_Custom(
        pretrained=pretrained,
        freeze=freeze,
        num_classes=num_classes,
        n_attributes=n_attributes,
        bottleneck=False,
        connect_CY=connect_CY,
        three_class=three_class
    )
