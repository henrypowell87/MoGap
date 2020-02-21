from torch import sqrt
from torch.nn import MSELoss, Module

class RMSE_loss(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = sqrt(self.mse(yhat, y) + self.eps)
        return loss
