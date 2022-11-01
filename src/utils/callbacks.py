import os
import torch

class SaveBestModel:
    def __init__(self):
        self.best_valid_loss = float('-inf')
        
    def __call__(
            self, current_valid_loss, epoch, model, configurations
    ):
        if current_valid_loss > self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation loss: {self.best_valid_loss:.3f}")
            saving_name = f"{configurations.model_tag}_epoch_{epoch}_loss_{current_valid_loss}.pth"
            saving_path = os.path.join(
                configurations.checkpoints_path, saving_name
            )
            torch.save(model.state_dict(), saving_path)