"""
InceptionV3 Network modified from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
New changes: add softmax layer + option for freezing lower layers except fc
"""
import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from bcos_resnet import resnet18 as bcos_resnet18

__all__ = ['MLP', 'End2EndModel', 'ResNet18_Custom', 'resnet18_custom', 'BcosResNet18_Custom', 'bcos_resnet18_custom']

class End2EndModel(nn.Module):
    def __init__(self, model1, model2, use_relu=False, use_sigmoid=False, n_class_attr=2):
        """
        End-to-end Concept Bottleneck Model that connects:
            stage 1: X -> C (first_model)
            stage 2: C -> Y (sec_model)
        Works for both InceptionV3-based and ResNet18-based architectures.
        """
        super(End2EndModel, self).__init__()
        self.first_model = model1
        self.sec_model = model2
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid
        self.n_class_attr = n_class_attr

    def forward_stage2(self, stage1_out):
        """
        Process stage1 outputs (concepts) and feed them into stage2.
        """
        if isinstance(stage1_out, dict):
            # Handle dictionary outputs like {0: y_logits, 1: concept_logits}
            stage1_out = list(stage1_out.values())

        # Apply optional activations
        if self.use_relu:
            attr_outputs = [nn.ReLU()(o) for o in stage1_out]
        elif self.use_sigmoid:
            attr_outputs = [nn.Sigmoid()(o) for o in stage1_out]
        else:
            attr_outputs = stage1_out

        # Concatenate all concept predictions
        stage2_inputs = torch.cat(attr_outputs, dim=1)

        # Feed into second model (C->Y)
        all_out = [self.sec_model(stage2_inputs)]

        # Also return original concept predictions
        all_out.extend(stage1_out)
        return all_out

    def forward(self, x):
        """
        Handles both training (with aux_outputs) and inference (single output).
        """
        outputs = self.first_model(x)

        if isinstance(outputs, tuple) and len(outputs) == 2:
            # (outputs, aux_outputs) -> Inception-style
            outputs, aux_outputs = outputs
            return self.forward_stage2(outputs), self.forward_stage2(aux_outputs)
        else:
            # Only one output -> ResNet18 style
            return self.forward_stage2(outputs)
    
class BcosResNet18_Custom(nn.Module):
    def __init__(self, pretrained=True, freeze=False, num_classes=200,
                 n_attributes=None, bottleneck=False, connect_CY=False,
                 three_class=False, dropout_p=0.5):
        super(BcosResNet18_Custom, self).__init__()
        self.connect_CY = connect_CY
        self.bottleneck = bottleneck
        self.three_class = three_class
        self.n_attributes = n_attributes

        # Load pretrained B-cos ResNet-18
        self.resnet = bcos_resnet18(pretrained=pretrained, in_chans=6, num_classes=1)
        print("Has relprop:", hasattr(self.resnet, 'relprop'))
        self.resnet.fc = nn.Identity()

        # Optionally freeze backbone
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Get the output dimension of the last conv block
        in_features = self.resnet.num_features

        # Add dropout for regularization
        # self.dropout = nn.Dropout(p=dropout_p)

        # Concept prediction head (X → C)
        if n_attributes is not None:
            out_dim = n_attributes * (3 if three_class else 1)
            # self.concept_head = nn.Linear(in_features, out_dim)
            self.concept_head = nn.Sequential(
                nn.Linear(in_features, out_dim),
                nn.BatchNorm1d(out_dim),  # helps stabilize concept training
                nn.ReLU(),
                nn.Dropout(p=dropout_p)
            )
        else:
            self.concept_head = None

        # Classification head (C → Y) or (X → Y)
        if connect_CY and self.concept_head is not None:
            cls_input_dim = out_dim
            # self.classifier = nn.Linear(cls_input_dim, num_classes)
            self.classifier = nn.Sequential(
                nn.Linear(cls_input_dim, num_classes),
                # nn.Dropout(p=dropout_p)
            )
        else:
            # self.classifier = nn.Linear(in_features, num_classes)
            self.classifier = nn.Sequential(
                nn.Linear(in_features, num_classes),
                # nn.Dropout(p=dropout_p)
            )

    def forward(self, x):
        features = self.resnet.forward_features(x)
        pooled = self.resnet.avgpool(features)
        pooled = pooled.flatten(1)

        # pooled = self.dropout(pooled)  # Regularize pooled features

        outputs = {}
        if self.concept_head is not None:
            concepts = self.concept_head(pooled)
            outputs[1] = concepts

        if self.connect_CY and self.concept_head is not None:
            cls_input = concepts
        else:
            cls_input = pooled

        outputs[0] = self.classifier(cls_input)
        return outputs


# class ResNet18_Custom(nn.Module):
#     def __init__(self, pretrained=True, freeze=False, num_classes=200,
#                  n_attributes=None, bottleneck=False, connect_CY=False,
#                  three_class=False):
#         super(ResNet18_Custom, self).__init__()
#         self.connect_CY = connect_CY
#         self.bottleneck = bottleneck
#         self.three_class = three_class
#         self.n_attributes = n_attributes

#         # Load pretrained ResNet-18
#         self.resnet = models.resnet18(pretrained=pretrained)

#         # Optionally freeze feature extractor
#         if freeze:
#             for param in self.resnet.parameters():
#                 param.requires_grad = False

#         # Remove the default classifier head
#         in_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Identity()  # just feature extractor now

#         # If predicting attributes (multitask or CBM), define concept head
#         if n_attributes is not None:
#             # Concept prediction head (X → C)
#             if three_class:
#                 self.concept_head = nn.Linear(in_features, n_attributes * 3)
#             else:
#                 self.concept_head = nn.Linear(in_features, n_attributes)
#         else:
#             self.concept_head = None

#         # Classification head (either directly from features or via concepts)
#         if connect_CY and self.concept_head is not None:
#             # When concept predictions are used as input to classifier (X→C→Y)
#             cls_input_dim = n_attributes * (3 if three_class else 1)
#             self.classifier = nn.Linear(cls_input_dim, num_classes)
#         else:
#             # When predicting Y directly from image features (X→Y)
#             self.classifier = nn.Linear(in_features, num_classes)

#     def forward(self, x):
#         features = self.resnet(x)
#         outputs = {}

#         # Concept predictions (if defined)
#         if self.concept_head is not None:
#             concepts = self.concept_head(features)
#             outputs['concepts'] = concepts

#         # Classification prediction
#         if self.connect_CY and self.concept_head is not None:
#             # Use concept predictions for classification
#             cls_input = concepts
#         else:
#             # Use image features directly
#             cls_input = features

#         outputs['class'] = self.classifier(cls_input)

#         return outputs

class FC(nn.Module):

    def __init__(self, input_dim, output_dim, expand_dim, stddev=None):
        """
        Extend standard Torch Linear layer to include the option of expanding into 2 Linear layers
        """
        super(FC, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim > 0:
            self.relu = nn.ReLU()
            self.fc_new = nn.Linear(input_dim, expand_dim)
            self.fc = nn.Linear(expand_dim, output_dim)
        else:
            self.fc = nn.Linear(input_dim, output_dim)
        if stddev:
            self.fc.stddev = stddev
            if expand_dim > 0:
                self.fc_new.stddev = stddev

    def forward(self, x):
        if self.expand_dim > 0:
            x = self.fc_new(x)
            x = self.relu(x)
        x = self.fc(x)
        return x
        
# class BcosResNet18_Custom(nn.Module):
#     """
#     Drop-in replacement for Inception3, wrapping BcosResNet18_Custom.
#     Handles multiple outputs (attributes + main task), optional CY connection, etc.
#     """

#     def __init__(self, num_classes, n_attributes=0, bottleneck=False, expand_dim=0, three_class=False, connect_CY=False, aux_logits=False):
#         super(BcosResNet18_Custom, self).__init__()
#         self.n_attributes = n_attributes
#         self.bottleneck = bottleneck
#         self.connect_CY = connect_CY
#         self.aux_logits = aux_logits

#         # Base BcosResNet backbone
#         self.backbone = bcos_resnet18(
#             pretrained=True,
#             freeze=False,
#             num_classes=num_classes,
#             n_attributes=n_attributes,
#             connect_CY=connect_CY,
#             three_class=three_class,
#         )

#         # Final FCs like in Inception3
#         self.all_fc = nn.ModuleList()
#         if connect_CY:
#             self.cy_fc = FC(n_attributes, num_classes, expand_dim)
#         else:
#             self.cy_fc = None

#         if n_attributes > 0:
#             if not bottleneck:
#                 self.all_fc.append(FC(512, num_classes, expand_dim))
#             for _ in range(n_attributes):
#                 self.all_fc.append(FC(512, 1, expand_dim))
#         else:
#             self.all_fc.append(FC(512, num_classes, expand_dim))

#     def forward(self, x):
#         feats = self.backbone.features(x)  # assume backbone exposes features() or similar
#         feats = F.adaptive_avg_pool2d(feats, (1, 1))
#         feats = feats.view(feats.size(0), -1)

#         out = [fc(feats) for fc in self.all_fc]

#         if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
#             attr_preds = torch.cat(out[1:], dim=1)
#             out[0] += self.cy_fc(attr_preds)

#         if self.training and self.aux_logits:
#             return out, out  # mimic Inception3 aux head
#         return out

# def bcos_resnet18(pretrained, freeze, **kwargs):
#     """
#     Bcos-ResNet18 model replacement for InceptionV3.
#     Args:
#         pretrained (bool): Whether to use pretrained weights
#         freeze (bool): Whether to freeze lower layers (except fc)
#         kwargs: includes num_classes, n_attributes, expand_dim, bottleneck, connect_CY, etc.
#     """
#     model = BcosResNet18_Custom(
#         pretrained=pretrained,
#         freeze=freeze,
#         num_classes=kwargs.get("num_classes", 1000),
#         n_attributes=kwargs.get("n_attributes", 0),
#         connect_CY=kwargs.get("connect_CY", False),
#         three_class=kwargs.get("three_class", False),
#     )
#     return model

class ResNet18_Custom(nn.Module):
    def __init__(
        self,
        pretrained=False,
        freeze=False,
        num_classes=200,
        n_attributes=0,
        bottleneck=False,
        expand_dim=0,
        three_class=False,
        connect_CY=False,
    ):
        """
        Modified ResNet18 architecture for Concept Bottleneck Models.

        Args:
            pretrained (bool): If True, load pretrained ImageNet weights.
            freeze (bool): If True, freeze all convolutional layers.
            num_classes (int): Number of target classes (e.g., 200 for CUB-200).
            n_attributes (int): Number of concept attributes.
            bottleneck (bool): Whether to use the model as X->C bottleneck model.
            expand_dim (int): Expansion dimension for MLP or FC layers.
            three_class (bool): Whether attributes are 3-class (0/1/2).
            connect_CY (bool): Whether to connect C->Y inside the same model.
        """
        super(ResNet18_Custom, self).__init__()

        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.connect_CY = connect_CY
        self.three_class = three_class
        self.num_classes = num_classes

        # --- Load Base Model ---
        base_model = models.resnet18(pretrained=pretrained)
        in_features = base_model.fc.in_features

        # Optionally freeze layers
        if freeze:
            for name, param in base_model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False

        # Remove the original fully connected layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])

        # --- Classification heads ---
        all_fc = nn.ModuleList()

        if n_attributes > 0:
            # For multitask mode: predict both Y and concepts
            if not bottleneck:
                all_fc.append(nn.Linear(in_features, num_classes))
            for _ in range(n_attributes):
                if three_class:
                    all_fc.append(nn.Linear(in_features, 3))
                else:
                    all_fc.append(nn.Linear(in_features, 1))
        else:
            # For X->Y only
            all_fc.append(nn.Linear(in_features, num_classes))

        self.all_fc = all_fc

        # Optional connection: attribute predictions → final class prediction
        if connect_CY:
            input_dim = n_attributes * (3 if three_class else 1)
            self.cy_fc = nn.Linear(input_dim, num_classes)
        else:
            self.cy_fc = None

    def forward(self, x):
        # Base feature extraction
        x = self.features(x)
        x = torch.flatten(x, 1)

        # Collect outputs
        outputs = []
        for fc in self.all_fc:
            outputs.append(fc(x))

        # Connect C → Y if applicable
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(outputs[1:], dim=1)
            outputs[0] = outputs[0] + self.cy_fc(attr_preds)

        return outputs

def resnet18_custom(pretrained, freeze, **kwargs):
    """
    ResNet18 model architecture for Concept Bottleneck Models.

    Args:
        pretrained (bool): If True, load pretrained weights on ImageNet.
        freeze (bool): If True, freeze all convolutional layers.
        kwargs: Arguments for ResNet18_Custom
    """
    model = ResNet18_Custom(pretrained=pretrained, freeze=freeze, **kwargs)
    return model
    
# class MLP(nn.Module):
#     def __init__(self, input_dim, num_classes, expand_dim):
#         super(MLP, self).__init__()
#         self.expand_dim = expand_dim
#         if self.expand_dim:
#             self.linear = nn.Linear(input_dim, expand_dim)
#             self.activation = torch.nn.ReLU()
#             self.linear2 = nn.Linear(expand_dim, num_classes) #softmax is automatically handled by loss function
#         self.linear = nn.Linear(input_dim, num_classes)

#     def forward(self, x):
#         x = self.linear(x)
#         if hasattr(self, 'expand_dim') and self.expand_dim:
#             x = self.activation(x)
#             x = self.linear2(x)
#         return x

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, expand_dim=0):
        super(MLP, self).__init__()
        self.expand_dim = expand_dim

        if expand_dim and expand_dim > 0:
            self.linear1 = nn.Linear(input_dim, expand_dim)
            self.activation = nn.ReLU()
            self.linear2 = nn.Linear(expand_dim, num_classes)
        else:
            self.linear1 = nn.Linear(input_dim, num_classes)
            self.linear2 = None

    def forward(self, x):
        x = self.linear1(x)
        if self.linear2 is not None:
            x = self.activation(x)
            x = self.linear2(x)
        return x