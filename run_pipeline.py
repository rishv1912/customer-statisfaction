from pipelines.training_pipeline import training_pipeline
from zenml.client import Client


if __name__ == "__main__":
    # Run the pipelines
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path="/Users/sanjayprajapati/Documents/MLops/customer_statisfaction/data/olist_customers_dataset.csv")
    # file:/Users/sanjayprajapati/Library/Application Support/zenml/local_stores/e844322c-78fe-4c54-b999-e47c7263ddd6/mlruns

# mlflow ui --backend-store-ui "file:/Users/sanjayprajapati/Library/Application Support/zenml/local_stores/e844322c-78fe-4c54-b999-e47c7263ddd6/mlruns"