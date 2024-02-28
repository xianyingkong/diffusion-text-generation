from .transformer import TransformerNetModel
from .gaussian_diffusion import get_named_beta_schedule, SpacedDiffusion

def create_model_and_diffusion(
    hidden_t_dim,
    hidden_dim,
    vocab_size,
    config_name,
    use_plm_init,
    dropout,
    diffusion_steps,
    noise_schedule,
    predict_xstart,
    rescale_timesteps,
    **kwargs,
):
    model = TransformerNetModel(
        input_dims=hidden_dim,
        output_dims=hidden_dim,
        hidden_t_dim=hidden_t_dim,
        dropout=dropout,
        config_name=config_name,
        vocab_size=vocab_size,
        init_pretrained=use_plm_init
    )

    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)

    diffusion = SpacedDiffusion(
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
    )

    return model, diffusion