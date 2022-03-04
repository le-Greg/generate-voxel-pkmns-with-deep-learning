import torch


# Augmentations on the colored voxels
# Inspired by https://github.com/albumentations-team/albumentations
# and https://github.com/hiram64/3D-VoxelDataGenerator


def vx_flip(dim: int):
    """
    Flip one of the axis of the voxel grid
    For pkmn dataset convention, 0 = left-right, 1 = up-down, 2 = forward-backward
    """
    assert dim in [0, 1, 2]  # X, Y or Z

    def f(voxels):
        return torch.flip(voxels, dims=[dim + 1])

    return f


def vx_smallshift(dim: int):
    """
    Shift the progressive3dgan in on axis, as long as there is enough place to do so
    = no progressive3dgan in the outer layers
    For pkmn dataset convention, 0 = left-right, 1 = up-down, 2 = forward-backward
    """
    assert dim in [0, 1, 2]  # X, Y or Z

    def f(voxels):
        shifts = torch.nonzero(torch.sum(voxels[3], dim=[(dim + 1) % 3, (dim + 2) % 3]))
        shifts = list(range(shifts.min())) + list(range(shifts.max() + 1, voxels.shape[dim + 1]))
        if len(shifts) == 0:
            return voxels
        shift = shifts[torch.randint(0, len(shifts), [1])]
        shift = shift + 1 if shift < voxels.shape[dim + 1] // 2 else shift - voxels.shape[dim + 1]
        voxels = torch.roll(voxels, shifts=-shift, dims=dim + 1)
        return voxels

    return f


def vx_color_gaussian_noise(min_std=0.05, max_std=0.1):
    """
    Additive gaussian noise in the 3 RGB channels
    """
    def f(voxels):
        voxels[:3] = voxels[:3] + torch.randn(voxels[:3].shape, device=voxels.device) * \
                     (torch.rand([1], device=voxels.device) * (max_std - min_std) + min_std)
        return voxels

    return f


def vx_position_gaussian_noise(min_std=0.05, max_std=0.1):
    """
    Additive gaussian noise in the voxel channel
    """
    def f(voxels):
        voxels[3] = voxels[3] + torch.randn(voxels[3].shape, device=voxels.device) * \
                    (torch.rand([1], device=voxels.device) * (max_std - min_std) + min_std)
        return voxels

    return f


def vx_rgbshift(min_std=0.05, max_std=0.1):
    """
    Shift for each color channel (R=R+a, G=G+b, B=B+c)
    """
    def f(voxels):
        voxels[:3] = voxels[:3] + torch.randn([3, 1, 1, 1], device=voxels.device) * \
                     (torch.rand([1], device=voxels.device) * (max_std - min_std) + min_std)
        return voxels

    return f


def vx_gamma(max_gamma=0.1):
    """
    Gamma shift
    """
    def f(voxels):
        voxels = voxels.clamp(min=0)
        gamma = (-1 ** torch.randint(0, 1, [1])) * torch.rand([1]) * max_gamma
        voxels[:3] = torch.pow(voxels[:3], 1 + gamma.item())
        return voxels

    return f


def custom_augment(x, level: float = 1.):
    """
    Small presets functions, to control augmentation with a simple level parameter
    """
    if level > 0.3:
        for f in [vx_color_gaussian_noise(min_std=0., max_std=0.1 * level),
                  vx_rgbshift(min_std=0., max_std=0.1 * level),
                  vx_gamma(max_gamma=0.1 * level)]:
            x = f(x)

    for f in [vx_flip(0), vx_smallshift(1), vx_smallshift(2)]:
        if torch.rand([1]) <= level:
            x = f(x)
    if level > 0.3:
        x = vx_position_gaussian_noise(min_std=0., max_std=0.1 * level)(x)
    return x
