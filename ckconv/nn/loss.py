import torch
from ckconv.nn.ckconv import CKConvBase


class LnLoss(torch.nn.Module):
    def __init__(
        self,
        weight_loss: float,
        norm_type: int,
    ):
        """
        Computes the Ln loss on the convolutional layers of a CNN or a CKCNN
        :param weight_loss: Specifies the weight with which the loss will be summed to the total loss.
        :param norm_type: Type of norm, e.g., 1 = L1 loss, 2 = L2 loss, ...
        """
        super().__init__()
        self.weight_loss = weight_loss
        self.norm_type = norm_type

    def _calculate_loss_weights(self, model):
        loss = 0.0
        # Go through non-CKConv modules
        params_outside_kernelnets = filter(
            lambda x: "Kernel" not in x[0], model.named_parameters()
        )
        for named_param in params_outside_kernelnets:
            loss += named_param[1].norm(self.norm_type)
        # Go through CKConv modules.
        # If the module is a CKConv, calculate on the sampled kernel
        for m in model.modules():
            if isinstance(m, CKConvBase):
                loss += m.conv_kernel.norm(self.norm_type)
        return loss

    def forward(
        self,
        model: torch.nn.Module,
    ):
        # Calculate loss
        loss = self._calculate_loss_weights(model)
        # Re-weight by self.weight_loss
        loss = self.weight_loss * loss
        return loss
