import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    project="Karachi_AQI_Predictor_1"
)

fs = project.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="karachi_aqi_features",
    version=1,
    description="Daily Karachi AQI prediction features",
    primary_key=["date"],
    event_time="date"
)
FEATURE_GROUP_NAME = "aqi_features_v2"
FEATURE_GROUP_VERSION = 1

fg = fs.get_or_create_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
    description="AQI features for 3-day AQI prediction",
    primary_key=["date"],
    event_time="date",
    online_enabled=False
)

print("Feature Group ready:")
print(fg.name)
print("Version:", fg.version)
print("\nFeature Group schema:")

for feature in fg.features:
    print(
        feature.name,
        "->",
        feature.type
    )