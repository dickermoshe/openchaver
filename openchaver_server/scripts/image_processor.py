from camera.models import ScreenCapture
import time
from django.conf import settings
from .utils import get_size_of_screenshot_folder, get_current_cpu_usage
import logging
from django.utils import timezone
logger = logging.getLogger(__name__)

def run():
    """
    Take a screen capture and process it.
    """
    while True:
        for image in ScreenCapture.objects.filter(is_proccessed=False):
            # Delay 5 seconds if the CPU is over XX% usage and the screenshot folder is not full.
            if (get_size_of_screenshot_folder() < settings.MAX_SCREENSHOT_SIZE
                    and get_current_cpu_usage() > settings.MAX_CPU_PERCENT):
                time.sleep(5)
            image.process_image()
        
        images = ScreenCapture.objects.all()
        for image in images:
            # If the image is older than a month, delete it.
            if image.taken_at < timezone.now()-timezone.timedelta(days=30):
                image.delete()
            
            # If the image nsfw average is 0, delete it.
            elif image.is_proccessed and image.average_nsfw < settings.REQ_NSFW_4_DELETION:
                image.delete()
        
        for i in range(1,30):
            day_images = ScreenCapture.objects.filter(taken_at__date=timezone.now() - timezone.timedelta(days=i)).order_by('-average_nsfw')
            if len(day_images) > 10:
                for image in day_images[10:]:
                    image.delete()

        time.sleep(30)