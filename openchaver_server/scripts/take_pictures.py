from camera.models import ScreenCapture
import time
from random import randint
def run():
    """
    Take a screen capture and process it.
    """
    while True:
        ScreenCapture.snap(full_monitor=False)
        time.sleep(randint(5,10))