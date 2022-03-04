import torch
from ..nvae3d.training_functions import AvgrageMeter, vae_terms


def cross_entropy_normal(all_eps):
    from ..nvae3d.distributions import log_p_standard_normal

    cross_entropy = 0.
    neg_log_p_per_group = []
    for eps in all_eps:
        neg_log_p_conv = - log_p_standard_normal(eps)
        neg_log_p = torch.sum(neg_log_p_conv, dim=[1, 2, 3, 4])
        cross_entropy += neg_log_p
        neg_log_p_per_group.append(neg_log_p_conv)

    return cross_entropy, neg_log_p_per_group


def sum_log_q(all_log_q):
    log_q = 0.
    for log_q_conv in all_log_q:
        log_q += torch.sum(log_q_conv, dim=[1, 2, 3, 4])

    return log_q


def get_mixed_prediction(mixed_prediction, param, mixing_logit, mixing_component=None):
    if mixed_prediction:
        assert mixing_component is not None, 'Provide mixing component when mixed_prediction is enabled.'
        coeff = torch.sigmoid(mixing_logit)
        param = (1 - coeff) * mixing_component + coeff * param

    return param


def kl_per_group_vada(all_log_q, all_neg_log_p):
    assert len(all_log_q) == len(all_neg_log_p)

    kl_all_list = []
    kl_diag = []
    for log_q, neg_log_p in zip(all_log_q, all_neg_log_p):
        kl_diag.append(torch.mean(torch.sum(neg_log_p + log_q, dim=[2, 3, 4]), dim=0))
        kl_all_list.append(torch.sum(neg_log_p + log_q, dim=[1, 2, 3, 4]))

    # kl_all = torch.stack(kl_all, dim=1)   # batch x num_total_groups
    kl_vals = torch.mean(torch.stack(kl_all_list, dim=1), dim=0)   # mean per group

    return kl_all_list, kl_vals, kl_diag


def infer_active_variables(train_queue, vae, max_iter=None):
    kl_meter = AvgrageMeter()
    vae.eval()
    for step, x in enumerate(train_queue):
        if max_iter is not None and step > max_iter:
            break

        x = x.to(vae.device)

        # apply vae:
        with torch.set_grad_enabled(False):
            _, all_log_q, all_eps = vae(x)
            all_eps = vae.concat_eps_per_scale(all_eps)
            all_log_q = vae.concat_eps_per_scale(all_log_q)
            log_q, log_p, kl_all, kl_diag = vae_terms(all_log_q, all_eps)
            kl_meter.update(kl_diag[0], 1)  # only the top scale

    return kl_meter.avg > 0.1


def trace_df_dx_hutchinson(f, x, noise, no_autograd):
    """
    Hutchinson's trace estimator for Jacobian df/dx, O(1) call to autograd
    """
    if no_autograd:
        # the following is compatible with checkpointing
        torch.sum(f * noise).backward()
        # torch.autograd.backward(tensors=[f], grad_tensors=[noise])
        jvp = x.grad
        trJ = torch.sum(jvp * noise, dim=[1, 2, 3, 4])
        x.grad = None
    else:
        jvp = torch.autograd.grad(f, x, noise, create_graph=False)[0]
        trJ = torch.sum(jvp * noise, dim=[1, 2, 3, 4])
        # trJ = torch.einsum('bijk,bijk->b', jvp, noise)  # we could test if there's a speed difference in einsum vs sum

    return trJ


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape, device=y.device) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


def vae_regularization(train_vae, weight_decay_norm_vae, vae_sn_calculator):
    regularization_q, vae_norm_loss, vae_bn_loss, vae_wdn_coeff = 0., 0., 0., weight_decay_norm_vae
    if train_vae:
        vae_norm_loss = vae_sn_calculator.spectral_norm_parallel()
        vae_bn_loss = vae_sn_calculator.batchnorm_loss()
        regularization_q = (vae_norm_loss + vae_bn_loss) * vae_wdn_coeff

    return regularization_q, vae_norm_loss, vae_bn_loss, vae_wdn_coeff


def dae_regularization(weight_decay_norm_dae, dae_sn_calculator):
    dae_wdn_coeff = weight_decay_norm_dae
    dae_norm_loss = dae_sn_calculator.spectral_norm_parallel()
    dae_bn_loss = dae_sn_calculator.batchnorm_loss()
    regularization_p = (dae_norm_loss + dae_bn_loss) * dae_wdn_coeff

    return regularization_p, dae_norm_loss, dae_bn_loss, dae_wdn_coeff


def update_lr(learning_rate_dae, learning_rate_vae, train_vae, global_step, warmup_iters, dae_optimizer, vae_optimizer):
    if global_step < warmup_iters:
        lr = learning_rate_dae * float(global_step) / warmup_iters
        for param_group in dae_optimizer.param_groups:
            param_group['lr'] = lr

        if train_vae:
            lr = learning_rate_vae * float(global_step) / warmup_iters
            for param_group in vae_optimizer.param_groups:
                param_group['lr'] = lr
