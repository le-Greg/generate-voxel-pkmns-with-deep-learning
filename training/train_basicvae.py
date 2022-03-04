import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils_for_visualization import format_and_show_voxels


def train_basicvae(
        dataset,  # Voxel dataset, or a dataset with set_size and set_noise_level
        model,
        writer,  # Tensorboard writer
        lr=0.005,  # generator learning rate
        batch_size=16,  # Batch sizes for each voxel resolution
        n_epochs=10,  # Number of samples shown to the network
        device="cuda:0",  # Device
        noise_coeff=3,  # Evolution of noise following 1-x**noise_coeff, x between 0 (start) and 1 (end)
        samples_per_log=1000,  # Frequency of saving images
        weight_decay=0.0,
        kld_weight=0.00025,
        scheduler_gamma=0.95,

):
    assert dataset.n_cubes == 32

    model = model.to(device)

    # Setup Adam optimizers for both G and D
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)

    timestamp = 0
    time_to_log_imgs = 0

    # Loop through data
    for epoch in range(n_epochs):
        noise_level = 1. - (epoch / (n_epochs - 1)) ** noise_coeff
        dataset.set_noise_level(noise_level)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

        writer.add_scalar('BasicVAE/LR', optimizer.param_groups[0]["lr"], timestamp)
        writer.add_scalar('BasicVAE/Noise_level', noise_level, timestamp)

        for step, x in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", total=len(dataloader) - 1):
            real = x.to(device)
            current_batch_size = real.shape[0]

            recons, initial, mu, log_var = model(real)
            train_loss = model.loss_function(recons, initial, mu, log_var, kld_weight)

            loss = train_loss['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('BasicVAE/Loss', loss, timestamp)

            if timestamp >= time_to_log_imgs:

                format_and_show_voxels(real, save_as_file_on=writer,
                                       writer_params=('BasicVAE/Real', timestamp), image_size=256,
                                       packed=True)

                format_and_show_voxels(recons, save_as_file_on=writer,
                                       writer_params=('BasicVAE/Recons', timestamp), image_size=256,
                                       packed=True)

                gen = model.sample(num_samples=min(batch_size, 16), current_device=device)

                format_and_show_voxels(gen, save_as_file_on=writer,
                                       writer_params=('BasicVAE/Fake', timestamp), image_size=256,
                                       packed=True)

                while timestamp >= time_to_log_imgs:
                    time_to_log_imgs += samples_per_log

            timestamp += current_batch_size

        scheduler.step()

    return model
