import warnings

from utils.data_utils import extract_dataset, load_datasets, visualize_dataset
from model.model import build_model
from model.training import compile_model, evaluate_model, plot_history, train_model

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    extract_dataset()
    visualize_dataset()

    train_ds, validation_ds = load_datasets()

    model = build_model()
    model = compile_model(model)
    history = train_model(model, train_ds, validation_ds)

    plot_history(history)
    evaluate_model(model, validation_ds)
