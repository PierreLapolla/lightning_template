from time import perf_counter

from lightning.pytorch import seed_everything

from src.config import config
from src.helpers.decorators import timer
from src.helpers.menu import Menu
from src.helpers.torch_inference import TorchInference
from src.model_training.data_module import DataModule
from src.model_training.lightning_manager import lightning_manager
from src.model_training.mlp import MLP


def benchmark_inference():
    mlp_inference = TorchInference(MLP, 'logs/models/model_2024_07_29__16_06_16.pt')
    data_module = DataModule()
    data_module.setup('test')
    data_loader_length = len(data_module.test_dataloader())

    start_time = perf_counter()
    for x, _ in data_module.test_dataloader():
        _ = mlp_inference(x)

    average_time = (perf_counter() - start_time) / data_loader_length
    print(f"Average time taken over {data_loader_length} passes: {average_time * data_loader_length:.6f} ms")


@timer
def main() -> None:
    seed_everything(config.seed, workers=True)
    Menu({
        "1": ("Train model", lightning_manager.train_model),
        "2": ("Start sweep", lightning_manager.start_sweep),
        "3": ("Benchmark inference", benchmark_inference),
    }).start(timeout=60)


if __name__ == '__main__':
    main()
