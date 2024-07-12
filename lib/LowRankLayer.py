import torch
import torch.nn as nn

class LowRankLayer(nn.Module):
    """given a linear layer find low rank decomposition"""
    def __init__(self, desired_rank, weight, require_grad=True):
        super().__init__()
        self.rank = desired_rank
        
        
        results = torch.svd(weight)
        U = results[0][:, :desired_rank]
        S = results[1][:desired_rank]
        V = results[2][:, :desired_rank]

        self.U = nn.Linear(desired_rank, U.shape[0], bias=False).to(weight.device)
        self.V = nn.Linear(V.shape[0], desired_rank, bias=False).to(weight.device)

        self.U.weight.data = U.mul(S.sqrt()).to(torch.bfloat16).contiguous()
        self.V.weight.data = V.t().mul(S.sqrt().view(-1, 1)).to(torch.bfloat16).contiguous()

        if require_grad == False:
            self.U.weight.requires_grad = False
            self.V.weight.requires_grad = False
        else:
            self.U.weight.requires_grad = True
            self.V.weight.requires_grad = True
    
    def forward(self, x):
        output = self.U(self.V(x.to(torch.bfloat16)))
        return output


class LowRankLayerEval(nn.Module):
    """given a linear layer find low rank decomposition"""
    def __init__(self, desired_rank, weight, require_grad=True):
        super().__init__()
        self.rank = desired_rank
        self.U = nn.Linear(desired_rank, weight.shape[0], bias=False, dtype=torch.bfloat16).to(weight.device)
        self.V = nn.Linear(weight.shape[1], desired_rank, bias=False, dtype=torch.bfloat16).to(weight.device)

        if require_grad == False:
            self.U.weight.requires_grad = False
            self.V.weight.requires_grad = False
        else:
            self.U.weight.requires_grad = True
            self.V.weight.requires_grad = True
    


    def forward(self, x):
        output = self.U(self.V(x.to(torch.bfloat16)))
        return output

