from camera.models import ScreenCapture
from django.utils import timezone
import time
from django.conf import settings
def run():
    """
    Clean up images.
    """
    while True:
        images = ScreenCapture.objects.all()
        for image in images:
            # If the image is older than a month, delete it.
            if image.taken_at < timezone.now()-timezone.timedelta(days=30):
                image.delete()
            
            # If the image nsfw average is 0, delete it.
            elif image.average_nsfw > settings.REQ_NSFW_4_DELETION and image.is_proccessed:
                image.delete()
        
        for i in range(30):
            day_images = ScreenCapture.objects.filter(taken_at__date=timezone.now() - timezone.timedelta(days=i)).order_by('-average_nsfw')
            if len(day_images) > 10:
                for image in day_images[10:]:
                    image.delete()

        time.sleep(30)
