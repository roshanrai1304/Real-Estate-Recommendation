from pipelines.training_pipelines import train_pipelines
from zenml.client import Client


if __name__ == "__main__":
    # run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipelines(data_path=r"C:\Users\HP\Documents\mihir project\Real-Estate-Recommendation\data\processed_real_estate_data -  newLabel.csv")