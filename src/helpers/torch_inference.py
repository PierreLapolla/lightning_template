import torch
import numpy as np
from pathlib import Path
from typing import Union, Type


class TorchInference:
    def __init__(self, model_class: Type[torch.nn.Module], model_path: Union[str, Path], device: str = 'cuda') -> None:
        """
        Generic class for loading and using a PyTorch model for inference.
        :param model_class: class of the model to be loaded
        :param model_path: path to the model file
        :param device: what device to use for inference, 'cpu' or 'cuda' (default)
        """
        self.model: Union[torch.nn.Module, None] = None
        self.model_class = model_class
        self.model_path: Path = Path(model_path)
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.load_model()

    def load_model(self) -> None:
        self.model = self.model_class().to(self.device)
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"Successfully loaded model from {self.model_path}, using device: {self.device}")
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {e}")

    def __call__(self, x: torch.Tensor, to_numpy: bool = True) -> Union[torch.Tensor, np.ndarray]:
        """
        Perform inference on a batch of data.
        :param x: input data
        :param to_numpy: whether to return the output as a numpy array
        :return: model output
        """
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            return output.cpu().numpy() if to_numpy else output.cpu()
