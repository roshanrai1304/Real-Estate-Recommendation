from pipelines.training_pipelines import train_pipelines
import os


if __name__ == "__main__":
    # run the pipeline
    train_pipelines(data_path=f"data/{os.listdir('data')[0]}")