# main.py

# import logging
# from src import logger  # Triggers configure_logger() from __init__.py

# Get a module-level logger
# log = logging.getLogger(__name__)

# log.debug("This is a debug message.")
# log.info("This is an info message.")
# log.warning("This is a warning message.")
# log.error("This is an error message.")
# log.critical("This is a critical message.")


# # below code is to check the exception config
from src.logger import logging  #since in logger.py file we have already imported logging therefore here we are calling logging from there.we could also directly import logging as we did in above code
# from src import logger
from src import exception
import sys
log = logging.getLogger(__name__)
try:
    a = 1+'Z'
except Exception as e:
    log.info(e)
    raise exception.CustomException(e, sys) from e