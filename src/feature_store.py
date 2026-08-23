from hopsworks_connection import (
    get_project,
    get_feature_store,
    get_model_registry
)


project = get_project()

# print("Project:", project.name)

# fs = get_feature_store()

# print("Feature Store connected!")

# mr = get_model_registry()

# print("Model Registry connected!")