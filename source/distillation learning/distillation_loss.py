import torch
import torch.nn as nn

class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs_student, outputs_teacher):
        # Apply temperature scaling to student model's outputs
        outputs_student = outputs_student / self.temperature
        # Compute distillation loss
        loss = self.loss_fn(torch.log_softmax(outputs_student, dim=1), torch.softmax(outputs_teacher, dim=1))
        return loss