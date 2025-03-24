from loguru import logger
import requests
from churn_classification_engine.config import settings

DATA_PATH = settings.data_dir / settings.data_filename

if __name__ == "__main__":
    # Download the CSV file
    logger.info(f"Downloading CSV from {settings.data_url}")
    response = requests.get(settings.data_url)

    # Save the CSV file
    if response.status_code == 200:
        with open(DATA_PATH, "wb") as file:
            file.write(response.content)
        logger.success(f"CSV file downloaded at {DATA_PATH}")
    else:
        logger.error(f"Failed to download CSV. Status code: {response.status_code}")
