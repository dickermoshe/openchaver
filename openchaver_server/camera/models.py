
from django.db import models
from django.db.models.signals import post_delete ,pre_save
from django.dispatch import receiver

from io import BytesIO
from django.core.files.images import ImageFile
from PIL.Image import Image as PILImage
from PIL import Image
import uuid
from openchaver_server.settings import nsfw

def pillow_to_image_field(image:PILImage) -> ImageFile:
    """
    Converts a PILImage to an ImageField.
    """
    image_io = BytesIO()
    image.save(image_io, format='JPEG')
    filename = str(uuid.uuid4()) + '.jpg'
    return ImageFile( image_io, filename)

class ScreenCapture(models.Model):
    image = models.ImageField(upload_to='screenCaptures/')
    taken_at = models.DateTimeField(auto_now_add=True)
    input_name = models.TextField(max_length=500, default="")
    is_proccessed = models.BooleanField(default=False)
    is_parsed = models.BooleanField(default=False)
    is_thresholded = models.BooleanField(default=False)

    skin_percentage = models.FloatField(blank=True,null=True)

    average_safe = models.FloatField(blank=True,null=True)

    
    @staticmethod
    def snap(full_monitor = False):
        if full_monitor:
            shots = nsfw.camera.take_screenshot_of_monitor()
        else:
            shots = nsfw.camera.take_screenshot_of_active_window()

        for img in shots['images']:
            # Convert from pillow to imagefield
            img = pillow_to_image_field(img)
            ScreenCapture.objects.create(image=img,input_name=shots['title'])
    
    def process_image(self,skin_threshold=0,parse_images=True):
        """
        Convert a raw screen capture to a processed screen capture.
        """
        # Convert the raw screen capture to a PIL image
        image = Image.open(self.image.path)

        # Check if the image is NSFW
        results = nsfw.check_image(
            image,
            skin_threshold=skin_threshold,
            parse_images=parse_images
            )
        if len(results) == 0:
            self.is_proccessed = True
            self.save()
            return
        # Of all the images that passed the skin threshold, create a skin percentage of those images
        skin_pixel_counts = [i['skin_rating']['skin_pixel_count'] for i in results if i['nsfw_rating']]
        total_pixel_count = [i['skin_rating']['total_pixel_count'] for i in results if i['nsfw_rating']]
        is_safe = [i['nsfw_rating']['safe'] for i in results if i['nsfw_rating']]

        # # Get the worst image
        # worst = 0
        # worst_image = None
        # for i in results:
        #     if i['nsfw_rating']:
        #         if i['nsfw_rating']['unsafe'] > worst:
        #             worst = i['nsfw_rating']['unsafe']
        #             worst_image = i
        # print(worst_image)
        # worst_image['image'].show()


        try:
            self.skin_percentage = sum(skin_pixel_counts)/sum(total_pixel_count)
        except ZeroDivisionError:
            self.skin_percentage = 0
        
        try:
            self.average_safe = sum(is_safe)/len(is_safe)
        except ZeroDivisionError:
            self.average_safe = 0

        self.is_proccessed = True
        self.is_parsed = parse_images
        self.is_thresholded = skin_threshold > 0
        self.save()



        

            











            


