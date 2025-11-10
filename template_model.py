"""
Unified CBM architectures — InceptionV3/ResNet18/BcosResNet18
BcosResNet18_Custom built to mirror ResNet18_Custom structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from bcos_resnet import resnet18 as bcos_resnet18

__all__ = [
    'MLP',
    'End2EndModel',
    'ResNet18_Custom',
    'resnet18_custom',
    'BcosResNet18_Custom',
    'bcos_resnet18_custom'
]

# ================================
#  End-to-End Concept Bottleneck Model
# ================================
class End2EndModel(nn.Module):
    def __init__(self, model1, model2, use_relu=False, use_sigmoid=False, n_class_attr=2):
        """
        End-to-end Concept Bottleneck Model that connects:
            stage 1: X -> C (first_model)
            stage 2: C -> Y (sec_model)
        Works for both InceptionV3-based and ResNet18-based architectures.
        """
        super().__init__()
        self.first_model = model1
        self.sec_model = model2
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid
        self.n_class_attr = n_class_attr

    def forward_stage2(self, stage1_out):
        if isinstance(stage1_out, dict):
            stage1_out = list(stage1_out.values())

        if self.use_relu:
            attr_outputs = [F.relu(o) for o in stage1_out]
        elif self.use_sigmoid:
            attr_outputs = [torch.sigmoid(o) for o in stage1_out]
        else:
            attr_outputs = stage1_out

        stage2_inputs = torch.cat(attr_outputs, dim=1)
        all_out = [self.sec_model(stage2_inputs)]
        all_out.extend(stage1_out)
        # print(f'Outputs : {all_out}')
        return all_out

    def forward(self, x):
        outputs = self.first_model(x)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            outputs, aux_outputs = outputs
            return self.forward_stage2(outputs), self.forward_stage2(aux_outputs)
        else:
            return self.forward_stage2(outputs)


# ================================
#  FC / MLP Utilities
# ================================
class FC(nn.Module):
    def __init__(self, input_dim, output_dim, expand_dim=0):
        super().__init__()
        self.expand_dim = expand_dim
        if expand_dim > 0:
            self.fc1 = nn.Linear(input_dim, expand_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(expand_dim, output_dim)
        else:
            self.fc1 = nn.Linear(input_dim, output_dim)
            self.fc2 = None

    def forward(self, x):
        x = self.fc1(x)
        if self.fc2 is not None:
            x = self.relu(x)
            x = self.fc2(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, expand_dim=0):
        super().__init__()
        self.expand_dim = expand_dim
        if expand_dim > 0:
            self.linear1 = nn.Linear(input_dim, expand_dim)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(expand_dim, num_classes)
        else:
            self.linear1 = nn.Linear(input_dim, num_classes)
            self.linear2 = None

    def forward(self, x):
        x = self.linear1(x)
        if self.linear2 is not None:
            x = self.relu(x)
            x = self.linear2(x)
        return x


# ================================
#  Standard ResNet18 Backbone
# ================================
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
        super().__init__()

        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.connect_CY = connect_CY
        self.three_class = three_class
        self.num_classes = num_classes

        base_model = models.resnet18(pretrained=pretrained)
        in_features = base_model.fc.in_features

        # Optionally freeze all layers except final FCs
        if freeze:
            for name, param in base_model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

        self.features = nn.Sequential(*list(base_model.children())[:-1])

        all_fc = nn.ModuleList()

        if n_attributes > 0:
            if not bottleneck:
                all_fc.append(FC(in_features, num_classes, expand_dim))
            for _ in range(n_attributes):
                out_dim = 3 if three_class else 1
                all_fc.append(FC(in_features, out_dim, expand_dim))
        else:
            all_fc.append(FC(in_features, num_classes, expand_dim))

        self.all_fc = all_fc

        if connect_CY:
            input_dim = n_attributes * (3 if three_class else 1)
            self.cy_fc = FC(input_dim, num_classes, expand_dim)
        else:
            self.cy_fc = None

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        outputs = [fc(x) for fc in self.all_fc]

        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(outputs[1:], dim=1)
            outputs[0] = outputs[0] + self.cy_fc(attr_preds)

        return outputs


def resnet18_custom(pretrained, freeze, **kwargs):
    return ResNet18_Custom(pretrained=pretrained, freeze=freeze, **kwargs)


# ================================
#  Bcos-ResNet18 Backbone
# ================================
class BcosResNet18_Custom(nn.Module):
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
        BcosResNet18 version of ResNet18_Custom
        """
        super().__init__()

        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.connect_CY = connect_CY
        self.three_class = three_class
        self.num_classes = num_classes

        # Load pretrained Bcos-ResNet18 backbone
        self.backbone = bcos_resnet18(pretrained=pretrained, in_chans=6, num_classes=1)
        in_features = self.backbone.num_features

        # Optionally freeze early layers
        if freeze:
            for name, param in self.backbone.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

        self.features = nn.Sequential(*list(self.backbone.children())[:-1])

        all_fc = nn.ModuleList()

        if n_attributes > 0:
            if not bottleneck:
                all_fc.append(FC(in_features, num_classes, expand_dim))
            for _ in range(n_attributes):
                out_dim = 3 if three_class else 1
                all_fc.append(FC(in_features, out_dim, expand_dim))
        else:
            all_fc.append(FC(in_features, num_classes, expand_dim))

        self.all_fc = all_fc

        if connect_CY:
            input_dim = n_attributes * (3 if three_class else 1)
            self.cy_fc = FC(input_dim, num_classes, expand_dim)
        else:
            self.cy_fc = None

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        feats = torch.flatten(feats, 1)

        outputs = [fc(feats) for fc in self.all_fc]

        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(outputs[1:], dim=1)
            outputs[0] = outputs[0] + self.cy_fc(attr_preds)

        return outputs


def bcos_resnet18_custom(pretrained, freeze, **kwargs):
    """
    Factory function for BcosResNet18_Custom (CBM-compatible).
    """
    return BcosResNet18_Custom(pretrained=pretrained, freeze=freeze, **kwargs)
