from .transformer import TransformerNetModel
from .gaussian_diffusion import get_named_beta_schedule, SpacedDiffusion
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot(train_losses, val_losses):

    title_size = 28
    label_size = 21
    legend_size = 18
    tick_size = 17
    
    plot_folder_path = "plots"
    if not os.path.exists(plot_folder_path):
        os.makedirs(plot_folder_path)
        
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))  
    train_epochs = np.arange(1, len(train_losses)+1)
    val_epochs = np.arange(0, len(train_losses)+1, int(len(train_losses)/(len(val_losses)-1)))
    val_epochs[0] = 1
    
    ax.plot(train_epochs, train_losses, 'tab:blue', label="Training Loss")
    ax.plot(val_epochs, val_losses, 'tab:orange', label=f"Validation Loss (at every {int(len(train_losses)/(len(val_losses)-1))} interval)")
    ax.tick_params(labelsize=tick_size)
    ax.set_title('Loss', fontsize=title_size)
    ax.set_xlabel('Epochs', fontsize=label_size)
    ax.set_ylabel('Loss', fontsize=label_size)
    ax.legend(loc="upper right", fontsize=legend_size)
    

    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder_path, "loss_plot.png"))