import datetime
import logging
import os

PROJECT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')

LOG_DIR = os.path.join(PROJECT_FOLDER, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

CURRENT_DATE = datetime.datetime.now().strftime('%Y-%m-%d')

LOG_FILEPATH = os.path.join(LOG_DIR, f'log_{CURRENT_DATE}.txt')

# set logger and format
logger = logging.getLogger()
logFormat = logging.Formatter("[%(asctime)s - %(name)s]: %(levelname)s - %(message)s (%(funcName)s - line %(lineno)d)",
                              datefmt='%H:%M:%S')

# set level of logging
logger.setLevel(logging.INFO)

# file settings
fileHandler = logging.FileHandler(LOG_FILEPATH)
fileHandler.setFormatter(logFormat)
logger.addHandler(fileHandler)

