import os
import sys
import subprocess



PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)


def run_script(script_name):

    script_path = os.path.join(
        SRC_DIR,
        script_name
    )

    print("\n")
    print("=" * 60)
    print(f"RUNNING: {script_name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:

        print("\n")
        print("=" * 60)
        print(f"ERROR: {script_name} FAILED")
        print("=" * 60)

        sys.exit(result.returncode)

    print("\n")
    print("=" * 60)
    print(f"COMPLETED: {script_name}")
    print("=" * 60)


def main():

    print("\n")
    print("=" * 60)
    print("KARACHI AQI PREDICTION PIPELINE")
    print("=" * 60)

    print("\nPipeline steps:")

    print("1. Update daily data")
    print("2. Preprocess and create features")
    print("3. Train XGBoost model")
    print("4. Predict next 3 days AQI")
    run_script(
        "update_daily_data.py"
    )



    run_script(
        "preprocess_daily_data.py"
    )

    run_script(
        "model_training.py"
    )


    run_script(
        "predict.py"
    )


    print("\n")
    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nLatest prediction:")
    print(
        "results/latest_aqi_prediction.csv"
    )

if __name__ == "__main__":
    main()