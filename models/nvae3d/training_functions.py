import torch
import numpy as np


def kl_balancer_coeff(num_scales, groups_per_scale, fun):
    if fun == 'equal':
        coeff = torch.cat([torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0)
    elif fun == 'linear':
        coeff = torch.cat([(2 ** i) * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)],
                          dim=0)
    elif fun == 'sqrt':
        coeff = torch.cat(
            [np.sqrt(2 ** i) * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)],
            dim=0)
    elif fun == 'square':
        coeff = torch.cat(
            [np.square(2 ** i) / groups_per_scale[num_scales - i - 1] * torch.ones(groups_per_scale[num_scales - i - 1])
             for i in range(num_scales)], dim=0)
    else:
        raise NotImplementedError
    # convert min to 1.
    coeff /= torch.min(coeff)
    return coeff


def kl_per_group(kl_all):
    kl_vals = torch.mean(kl_all, dim=0)
    kl_coeff_i = torch.abs(kl_all)
    kl_coeff_i = torch.mean(kl_coeff_i, dim=0, keepdim=True) + 0.01

    return kl_coeff_i, kl_vals


def kl_balancer(kl_all, kl_coeff=1.0, kl_balance=False, alpha_i=None):
    if kl_balance and kl_coeff < 1.0:
        alpha_i = alpha_i.unsqueeze(0)

        kl_all = torch.stack(kl_all, dim=1)
        kl_coeff_i, kl_vals = kl_per_group(kl_all)
        total_kl = torch.sum(kl_coeff_i)

        kl_coeff_i = kl_coeff_i / alpha_i * total_kl
        kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
        kl = torch.sum(kl_all * kl_coeff_i.detach(), dim=1)

        # for reporting
        kl_coeffs = kl_coeff_i.squeeze(0)
    else:
        kl_all = torch.stack(kl_all, dim=1)
        kl_vals = torch.mean(kl_all, dim=0)
        kl = torch.sum(kl_all, dim=1)
        kl_coeffs = torch.ones(size=(len(kl_vals),))

    return kl_coeff * kl, kl_coeffs, kl_vals


def kl_coeff(step, total_step, constant_step, min_kl_coeff, max_kl_coeff):
    # return max(min(max_kl_coeff * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)
    return max(
        min(min_kl_coeff + (max_kl_coeff - min_kl_coeff) * (step - constant_step) / total_step, max_kl_coeff),
        min_kl_coeff
    )


def vae_terms(all_log_q, all_eps):
    from .distributions import log_p_standard_normal

    # compute kl
    kl_all = []
    kl_diag = []
    log_p, log_q = 0., 0.
    for log_q_conv, eps in zip(all_log_q, all_eps):
        log_p_conv = log_p_standard_normal(eps)
        kl_per_var = log_q_conv - log_p_conv
        kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3, 4]), dim=0))
        kl_all.append(torch.sum(kl_per_var, dim=[1, 2, 3, 4]))
        log_q += torch.sum(log_q_conv, dim=[1, 2, 3, 4])
        log_p += torch.sum(log_p_conv, dim=[1, 2, 3, 4])
    return log_q, log_p, kl_all, kl_diag


def reconstruction_loss(decoder, x, crop=False):
    from .distributions import DiscMixLogistic

    recon = decoder.log_p(x)
    if crop:
        recon = recon[:, :, 2:30, 2:30]

    if isinstance(decoder, DiscMixLogistic):
        return - torch.sum(recon, dim=[1, 2, 3])  # summation over RGB is done.
    else:
        return - torch.sum(recon, dim=[1, 2, 3, 4])


class AvgrageMeter(object):

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
