import numpy as np
import ot
import torch
from torch.autograd import Function

# loss function
def sigmoid_cross_entropy_loss(prediction, label):
    #print (label,label.max(),label.min())
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    #print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)


def cross_entropy_loss(prediction, label):
    #print (label,label.max(),label.min())
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    #print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)

def weighted_nll_loss(prediction, label):
    label = torch.squeeze(label.long(), dim=0)
    nch = prediction.shape[1]
    label[label >= nch] = 0
    cost = torch.nn.functional.nll_loss(prediction, label, reduce=False)
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.mul(cost, mask)
    return torch.sum(cost)

def weighted_cross_entropy_loss(prediction, label, output_mask=False):
    criterion = torch.nn.CrossEntropyLoss(reduce=False)
    label = torch.squeeze(label.long(), dim=0)
    nch = prediction.shape[1]
    label[label >= nch] = 0
    cost = criterion(prediction, label)
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask == 1] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.mul(cost, mask)
    if output_mask:
        return torch.sum(cost), (label != 0)
    else:
        return torch.sum(cost)

def l2_regression_loss(prediction, label, mask):
    label = torch.squeeze(label.float())
    prediction = torch.squeeze(prediction.float())
    mask = (mask != 0).float()
    num_positive = torch.sum(mask).float()
    cost = torch.nn.functional.mse_loss(prediction, label, reduce=False)
    cost = torch.mul(cost, mask)
    cost = cost / (num_positive + 0.00000001)
    return torch.sum(cost)
'''
class WassersteinLoss(Function):
    @staticmethod
    def forward(ctx, prediction, label, pre_idx, lab_idx, reg, numItermax=10000, stopThr=1e-7):
        # Compute the metric matrix
        XX = torch.einsum('ij,ij->i', (pre_idx, pre_idx)).unsqueeze(1)
        YY = torch.einsum('ij,ij->i', (lab_idx, lab_idx)).unsqueeze(0)
        distances = torch.einsum('ik,jk->ij', (pre_idx, lab_idx))
        distances = -2 * distances
        distances = distances + XX
        distances = distances + YY
        M = torch.sqrt(distances)
    
        # Compute Wasserstein Distance
        Nini = len(prediction)
        Nfin = len(label)
    
        # we assume that no distances are null except those of the diagonal of
        # distances
        u = torch.ones(Nini, dtype=torch.double).cuda() / Nini
        v = torch.ones(Nfin, dtype=torch.double).cuda() / Nfin
        
        #K= np.exp(-M/reg)
        K = torch.empty(M.shape, dtype=torch.double).cuda()
        torch.div(M, -reg, out=K)
        torch.exp(K, out=K)
    
        # print(np.min(K))        
        Kp = (1 / prediction).reshape(-1, 1) * K
        cpt = 0
        err = 1
        while (err > stopThr and cpt < numItermax):
            uprev = u
            vprev = v
    
            #KtransposeU = torch.mm(K.t(), u.unsqueeze(1)).reshape(-1)
            KtransposeU = torch.einsum('ij,i->j', (K, u))
            v = torch.div(label, KtransposeU)
            #u = 1. / torch.mm(Kp, v.unsqueeze(1)).reshape(-1)
            u = 1. / torch.einsum('ij,j->i', (Kp, v))
    
            if (torch.any(KtransposeU == 0) or
                    torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or
                    torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Warning: numerical errors at iteration', cpt)
                u = uprev
                v = vprev
                break
            if cpt % 10 == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations
                err = torch.sum((u - uprev)**2) / torch.sum((u)**2) + torch.sum((v - vprev)**2) / torch.sum((v)**2)
    
            cpt = cpt + 1
    
        # return only loss
        loss = torch.einsum('i,ij,j,ij', (u, K, v, M))
        # Save intermediate variables
        ctx.save_for_backward(u, K, torch.tensor([reg], dtype=torch.double).cuda())
        #ctx.save_for_backward(u)
        return loss.clone()
    
    @staticmethod    
    def backward(ctx, grad_output):
        u, K, reg = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        left = torch.mul(torch.log(u), reg)
        if torch.any(torch.isnan(left)) or torch.any(torch.isinf(left)):
            print('Error!')

        #Kinv = reg / K
        #right = torch.log(torch.sum(u) + 1e-10) * torch.sum(Kinv, dim=1)
        
        #return grad_input * (left - right), None, None, None, None, None, None 
        return grad_input * left, None, None, None, None, None, None
'''
class WassersteinLoss(Function):
    @staticmethod
    def forward(ctx, prediction, target, M, reg, numItermax=100, eps=1e-6):
        dim = prediction.size(1)          
        
        # Compute Wasserstein Distance
        u = torch.ones(1, dim, dtype=M.dtype).cuda()
        v = torch.ones(1, dim, dtype=M.dtype).cuda()
        
        #K= torch.exp((-M/reg)-1)
        K = torch.empty(M.shape, dtype=M.dtype).cuda()
        torch.div(M, -reg, out=K)
        K = K - 1
        torch.exp(K, out=K)
        
        #KM= K * M
        KM = torch.mul(K, M)
        
        #KlogK = K * logK
        KlogK = torch.mul(K, torch.log(K))    

        for i in range(numItermax):
            v = torch.div(target, torch.mm(u, K))
            u = torch.div(prediction, torch.mm(v, K.transpose(0, 1)))
            
        u[torch.abs(u) < eps] = eps
        v[torch.abs(v) < eps] = eps
            
        tmp1 = torch.mm(u, KM)
        loss = torch.mul(v, tmp1).sum()
        
        ulogu = torch.mul(u, torch.log(u))
        tmp2 = torch.mm(ulogu, K)
        entropy1 = torch.mul(tmp2, v).sum()

        vlogv = torch.mul(v, torch.log(v))
        tmp3 = torch.mm(vlogv, K.transpose(0, 1))
        entropy2 = torch.mul(tmp3, u).sum()

        tmp4 = torch.mm(u, KlogK)
        entropy3 = torch.mul(tmp4, v).sum()

        entropy = (entropy1 + entropy2 + entropy3) * reg
        loss_total = (loss + entropy)
            
        # Save intermediate variables
        ctx.save_for_backward(u, torch.tensor([reg], dtype=M.dtype).cuda())
        return loss_total.clone()
    
    @staticmethod    
    def backward(ctx, grad_output):
        u, reg = ctx.saved_tensors
        dim = u.size(1)
        grad_input = grad_output.clone()
        
        grad = torch.log(u) 
        shifting = torch.sum(grad, dim=1, keepdim=True) / dim

        return grad_input * (grad - shifting) * reg, None, None, None, None, None

def W_loss_lp(prediction, label, pre_idx, lab_idx):
    # Compute the metric matrix
    XX = torch.einsum('ij,ij->i', (pre_idx, pre_idx)).unsqueeze(1)
    YY = torch.einsum('ij,ij->i', (lab_idx, lab_idx)).unsqueeze(0)
    distances = torch.einsum('ik,jk->ij', (pre_idx, lab_idx))
    distances = -2 * distances
    distances = distances + XX
    distances = distances + YY
    M = torch.sqrt(distances)
    
    prediction = prediction.detach().cpu().numpy()
    label = label.cpu().numpy()
    M = M.cpu().numpy()    
    loss = ot.lp.emd2(prediction, label, M, numItermax=1000000)  
    
    return loss