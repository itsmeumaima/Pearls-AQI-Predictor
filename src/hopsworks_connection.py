import os

import hopsworks
from dotenv import load_dotenv

load_dotenv()


def get_project():

    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY is not set in .env"
        )

    project = hopsworks.login(
        api_key_value=api_key
    )

    return project


def get_feature_store():

    project = get_project()

    return project.get_feature_store()


def get_model_registry():

    project = get_project()

    return project.get_model_registry()