from camera.models import ScreenCapture
import time
from django.conf import settings
from .utils import get_size_of_screenshot_folder, get_current_cpu_usage
import logging
logger = logging.getLogger(__name__)

def run():
    """
    Take a screen capture and process it.
    """
    while True:
        for image in ScreenCapture.objects.filter(is_proccessed=False):
            # Delay 5 seconds if the CPU is over XX% usage and the screenshot folder is not full.
            if (get_size_of_screenshot_folder() < settings.MAX_SCREENSHOT_SIZE
                    or get_current_cpu_usage() > settings.MAX_CPU_PERCENT):
                time.sleep(5)
            image.process_image()
        time.sleep(30)