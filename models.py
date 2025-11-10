from template_model import MLP, End2EndModel, BcosResNet18_Custom, ResNet18_Custom
import torch.nn as nn


# Independent Model: X → C
def ModelXtoC(pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim, three_class):
    return BcosResNet18_Custom(
        pretrained=pretrained,
        freeze=freeze,
        num_classes=num_classes,
        n_attributes=n_attributes,
        bottleneck=True,
        expand_dim=expand_dim,
        three_class=three_class,
        connect_CY=False,
    )


# Oracle Model: C → Y
def ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim):
    if n_class_attr == 3:
        input_dim = n_attributes * n_class_attr
    else:
        input_dim = n_attributes
    return MLP(input_dim=input_dim, num_classes=num_classes, expand_dim=expand_dim)


# Sequential (two-stage) Model: X → Ĉ, Ĉ → Y
def ModelXtoChat_ChatToY(n_class_attr, n_attributes, num_classes, expand_dim):
    return ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim)


# Joint Model: X → C → Y (end-to-end)
def ModelXtoCtoY(n_class_attr, pretrained, freeze, num_classes, use_aux,
                 n_attributes, expand_dim, use_relu, use_sigmoid):
    model1 = BcosResNet18_Custom(
        pretrained=pretrained,
        freeze=freeze,
        num_classes=num_classes,
        n_attributes=n_attributes,
        bottleneck=True,
        expand_dim=expand_dim,
        three_class=(n_class_attr == 3),
        connect_CY=False,
    )

    if n_class_attr == 3:
        input_dim = n_attributes * n_class_attr
    else:
        input_dim = n_attributes

    model2 = MLP(input_dim=input_dim, num_classes=num_classes, expand_dim=expand_dim)

    return End2EndModel(model1, model2, use_relu, use_sigmoid, n_class_attr)


# Standard Model: X → Y only
def ModelXtoY(pretrained, freeze, num_classes, use_aux):
    return BcosResNet18_Custom(
        pretrained=pretrained,
        freeze=freeze,
        num_classes=num_classes,
        n_attributes=0,
        bottleneck=False,
        expand_dim=0,
        three_class=False,
        connect_CY=False,
    )


# Multitask Model: X → (C, Y)
def ModelXtoCY(pretrained, freeze, num_classes, use_aux,
               n_attributes, three_class, connect_CY):
    return BcosResNet18_Custom(
        pretrained=pretrained,
        freeze=freeze,
        num_classes=num_classes,
        n_attributes=n_attributes,
        bottleneck=False,
        expand_dim=0,
        three_class=three_class,
        connect_CY=connect_CY,
    )
