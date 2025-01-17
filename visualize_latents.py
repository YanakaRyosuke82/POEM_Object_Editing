 # Save intermediate results
        temp_dir = "output/temp_latents"
        os.makedirs(temp_dir, exist_ok=True)
        logging.info(f"Saving verifying intermediate results to {os.path.abspath(temp_dir)}...")

        step_size = max(len(latents_history) // 10, 1)
        for i in range(0, len(latents_history), step_size):
            latent = latents_history[i]
            
            # Save latent tensor
            torch.save(latent, os.path.join(temp_dir, f"latent_step_{i:04d}.pt"))
            
            # Visualize decoded image
            latent_viz = latent / 0.18215
            image = pipe.vae.decode(latent_viz).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            tvt.ToPILImage()(image[0]).save(
                os.path.join(temp_dir, f"latent_step_{i:04d}.png")
            )
            
            # Visualize latent channels
            latent_channels = latent_viz[0, :3]
            latent_channels = (latent_channels - latent_channels.min()) / (
                latent_channels.max() - latent_channels.min()
            )
            tvt.ToPILImage()(latent_channels).save(
                os.path.join(temp_dir, f"latent_channels_{i:04d}.png")
            )