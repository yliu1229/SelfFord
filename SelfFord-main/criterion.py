import torch
import torch.nn as nn

from utils import calc_topk_accuracy


def ce_loss(x, x_aug, temperature=1, lorentz=False, x0=0.6, gamma=0.04):
    B = x.size(0)
    N = 2 * B
    criterion = nn.CrossEntropyLoss()
    mask = mask_correlated_samples(B, x.device)

    z = torch.cat((x, x_aug), dim=0)

    sim = torch.matmul(z, z.T) / temperature
    sim_i_j = torch.diag(sim, B)
    sim_j_i = torch.diag(sim, -B)

    if lorentz:
        sim_i_j = 1/(1 + (sim_i_j - x0)**2/gamma)
        sim_j_i = 1/(1 + (sim_j_i - x0)**2/gamma)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N, device=x.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    loss = criterion(logits, labels)

    top1 = calc_topk_accuracy(logits, labels, (1,))
    # print('ce_logits = ', sim)

    return loss, top1


def mask_correlated_samples(batch_size, device):
    N = 2 * batch_size
    mask = torch.ones((N, N), device=device)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    mask = mask.bool()
    return mask


class my_criterion(nn.Module):
    def __init__(self, patch_num=2, temperature=1):
        super(my_criterion, self).__init__()
        self.patch_num = patch_num
        self.temperature = temperature

    def forward(self, results):
        # results = [(B,D), ...], len = patch_num + 1

        loss = 0
        # 1. For standard positive patch pairs, use InfoNCE loss
        loss1 = 0
        for i in range(self.patch_num-1):
            loss_tmp, _ = ce_loss(results[0], results[i+1], temperature=self.temperature)
            loss1 += loss_tmp
            print('\tloss1 = ', loss_tmp)
        loss += loss1 / (self.patch_num - 1)

        # 2. For weak positive patch paris, use Lorentz-InfoNCE loss
        loss2, _ = ce_loss(results[0], results[-1], temperature=self.temperature, lorentz=True)
        print('\tloss2 = ', loss2)
        loss += loss2

        return loss
