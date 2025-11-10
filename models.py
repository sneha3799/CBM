from template_model import MLP, End2EndModel, BcosResNet18_Custom, ResNet18_Custom
import torch.nn as nn


# Independent Model: X → C
def ModelXtoC(pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim, three_class):
    """
    Independent Concept Prediction Model: X → C

    Builds a model that maps raw input images (X) to concept predictions (C)
    without any connection to the downstream class prediction module (Y).

    Args:
        pretrained (bool): Whether to initialize the backbone (BcosResNet18_Custom) with pretrained weights.
        freeze (bool): Whether to freeze lower convolutional layers during training.
        num_classes (int): Number of output classes (used for compatibility; typically ignored here).
        use_aux (bool): Whether to use auxiliary classifiers (unused in this configuration).
        n_attributes (int): Number of concepts (attributes) to predict.
        expand_dim (int): Hidden layer expansion dimension for the concept bottleneck.
        three_class (bool): If True, predicts three-way concept classification (e.g., positive/negative/uncertain).

    Returns:
        nn.Module: A BcosResNet18_Custom model that predicts concepts from images.
    """
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
    """
    Oracle Classifier Model: C → Y

    Constructs a classifier that maps ground-truth concept vectors (C) directly to labels (Y).
    Used as an oracle to test the sufficiency of concept representations.

    Args:
        n_class_attr (int): Number of classes per attribute (e.g., 2 for binary, 3 for ternary concepts).
        n_attributes (int): Number of total attributes or concepts.
        num_classes (int): Number of target output classes.
        expand_dim (int): Optional hidden layer size in the MLP.

    Returns:
        nn.Module: An MLP mapping concept activations to class logits.
    """
    if n_class_attr == 3:
        input_dim = n_attributes * n_class_attr
    else:
        input_dim = n_attributes
    return MLP(input_dim=input_dim, num_classes=num_classes, expand_dim=expand_dim)


# Sequential (two-stage) Model: X → Ĉ, Ĉ → Y
def ModelXtoChat_ChatToY(n_class_attr, n_attributes, num_classes, expand_dim):
    """
    Two-Stage Sequential Model: X → Ĉ, Ĉ → Y

    Represents the second stage of a concept bottleneck pipeline, where
    predicted concepts (Ĉ) are used to infer class labels (Y).

    Args:
        n_class_attr (int): Number of classes per concept (e.g., 2 or 3).
        n_attributes (int): Number of total attributes or concepts.
        num_classes (int): Number of target output classes.
        expand_dim (int): Optional hidden layer size in the MLP.

    Returns:
        nn.Module: An MLP classifier that takes predicted concepts as input.
    """
    return ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim)


# Joint Model: X → C → Y (end-to-end)
def ModelXtoCtoY(n_class_attr, pretrained, freeze, num_classes, use_aux,
                 n_attributes, expand_dim, use_relu, use_sigmoid):
    """
    Joint End-to-End Model: X → C → Y

    Builds an end-to-end Concept Bottleneck Model that jointly learns both
    the concept predictor (X → C) and the label predictor (C → Y) in a single pipeline.

    Args:
        n_class_attr (int): Number of classes per concept (e.g., 2 for binary, 3 for ternary).
        pretrained (bool): Whether to initialize the backbone with pretrained weights.
        freeze (bool): Whether to freeze early layers in the backbone.
        num_classes (int): Number of target output classes.
        use_aux (bool): Whether to use auxiliary classifiers.
        n_attributes (int): Number of concepts to predict.
        expand_dim (int): Hidden dimension for MLP layers or intermediate expansion.
        use_relu (bool): Whether to apply ReLU activation between stages.
        use_sigmoid (bool): Whether to apply sigmoid activation on concept outputs.

    Returns:
        nn.Module: An End2EndModel combining concept prediction and class prediction.
    """
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
    """
    Standard Classification Model: X → Y

    Constructs a conventional image classification model (no concept bottleneck)
    that directly maps images to class predictions.

    Args:
        pretrained (bool): Whether to initialize the backbone with pretrained weights.
        freeze (bool): Whether to freeze feature extractor layers.
        num_classes (int): Number of target output classes.
        use_aux (bool): Whether to use auxiliary classifiers (unused here).

    Returns:
        nn.Module: A BcosResNet18_Custom backbone configured for direct classification.
    """
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
    """
    Multitask Model: X → (C, Y)

    Builds a multitask model that jointly predicts both concepts (C)
    and class labels (Y) from the same image features.

    Args:
        pretrained (bool): Whether to initialize the backbone with pretrained weights.
        freeze (bool): Whether to freeze lower layers during training.
        num_classes (int): Number of target output classes.
        use_aux (bool): Whether to use auxiliary classifiers.
        n_attributes (int): Number of concepts to predict.
        three_class (bool): If True, uses ternary concept prediction.
        connect_CY (bool): Whether to connect concept and class prediction layers.

    Returns:
        nn.Module: A BcosResNet18_Custom configured for multitask concept and label prediction.
    """
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
