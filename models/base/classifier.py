import torch
import torch.nn as nn

class Classifier1D(nn.Module):
    def __init__(self, inplanes=256, num_classes=2, dropout_rate=0.0):
        super(Classifier1D, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(inplanes, num_classes)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.avgpool(x.transpose(1,2))
        x = x.view(x.size(0), -1)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.fc(x)

        return x
    
class Classifier2D(nn.Module):
    def __init__(self, inplanes=256, num_classes=2, dropout_rate=0.0):
        super(Classifier2D, self).__init__()
        self.inplanes = inplanes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes, num_classes)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if x.size(1) == self.inplanes:
            x = self.avgpool(x)
        else:
            x = self.avgpool(x.permute(0, 3, 1, 2).contiguous())
        x = x.view(x.size(0), -1)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.fc(x)

        return x