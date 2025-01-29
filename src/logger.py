import logging
import os 
from datetime import datetime 
# create the log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# creating the path for the log file in current directory
logs_path = os.path.join(os.getcwd(),"logs", LOG_FILE)
# creating directory if it does not exist
os.makedirs(logs_path, exist_ok=True)
# final log path + file name 
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


logging.basicConfig(
    filename= LOG_FILE_PATH,
    format ="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)