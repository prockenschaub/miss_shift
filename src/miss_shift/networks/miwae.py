import torch
from torch import nn
import torch.distributions as td


def weights_init(layer) -> None:
    if type(layer) == nn.Linear:
        torch.nn.init.orthogonal_(layer.weight)


class MIWAE(nn.Module):
    def __init__(self, n_inputs, width, latent_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_inputs = n_inputs
        self.latent_size = latent_size

        self.p_z = td.Independent(
            td.Normal(
                loc=torch.zeros(self.latent_size),
                scale=torch.ones(self.latent_size),
            ),
            1,
        )

        self.encoder = nn.Sequential(
            torch.nn.Linear(n_inputs, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(
                width, 2 * latent_size
            ),  # the encoder will output both the mean and the diagonal covariance
        )

        self.decoder = nn.Sequential(
            torch.nn.Linear(latent_size, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(
                width, 2 * n_inputs
            ),  # the decoder will output both the mean and the diagonal covariance
        )

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

    def forward(self, iota_x, mask, L):
        out_encoder = self.encoder(iota_x)
        q_zgivenxobs = td.Independent(
            td.Normal(
                loc=out_encoder[..., : self.latent_size],
                scale=torch.nn.Softplus()(
                    out_encoder[..., self.latent_size : (2 * self.latent_size)]
                ),
            ),
            1,
        )

        zgivenx = q_zgivenxobs.rsample([L])
        zgivenx_flat = zgivenx.reshape([-1, self.latent_size])

        out_decoder = self.decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :self.n_inputs]
        all_scales_obs_model = (
            torch.nn.Softplus()(out_decoder[..., self.n_inputs : (2 * self.n_inputs)]) + 0.001
        )

        data_flat = torch.Tensor.repeat(iota_x, [L, 1]).reshape([-1, 1])
        tiledmask = torch.Tensor.repeat(mask, [L, 1])

        all_log_pxgivenz_flat = td.Normal(
            loc=all_means_obs_model.reshape([-1, 1]),
            scale=all_scales_obs_model.reshape([-1, 1])
        ).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([-1, self.n_inputs])

        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiledmask, 1).reshape(
            [L, -1]
        )
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        xgivenz = td.Independent(
            td.Normal(
                loc=all_means_obs_model,
                scale=all_scales_obs_model
            ),
            1,
        )

        return xgivenz, logpxobsgivenz, logpz, logq

