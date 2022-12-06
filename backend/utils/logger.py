# logger.py - Set up a4h logging (sending trace output to stdout)
# 1. In main script...
#     start_log(logging.INFO)  # Call this once
# 2. In specific module...
#     import logging
#     ...
#     log = logging.getLogger('a4h')
#     ...
#     log.info('Listening on port:'+str(port))
#     ...
#     log.debug('SomeVar='+str(SomeVar))

import logging


def start_log(level, full_meta=False):
    """Create a unified a4h logger"""

    # Create a single logger to be used by all
    new_log = logging.getLogger('a4h')
    new_log.setLevel(level)

    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('%(message)s [%(filename)s:%(lineno)d]'))
    new_log.addHandler(ch)

    if full_meta:
        # Add more context info to end of each log line
        ch.setFormatter(logging.Formatter(
            '"%(message)s"'
            + ',%(processName)s(%(process)d)'
            + ',%(threadName)s(%(thread)d)'
            + ',%(module)s:%(pathname)s:%(funcName)s:%(lineno)d'
            + ',%(asctime)s'
            + ',%(levelname)s')
        )
