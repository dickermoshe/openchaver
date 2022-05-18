import uuid
from io import BytesIO

from django.db import models
from django.core.files.images import ImageFile

from PIL.Image import Image as PILImage
from PIL import Image
import imagehash


from .nsfw import *

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
    average_nsfw = models.FloatField(blank=True,null=True)
    max_nsfw = models.FloatField(blank=True,null=True)

    
    @staticmethod
    def snap(full_monitor = False):
        if full_monitor:
            coord_list = get_coordinates_on_screen('monitor')
            title = 'Monitor'
        else:
            window = get_active_window()
            if window == None:
                return
            title = get_title_of_window(window)
            coord_list = get_coordinates_on_screen(window)
            

        for coord in coord_list:
            coord = fit_coordinates_to_monitor(coord)
            image = take_picture_of_coordinates(coord)
            if image == None:
                continue

            screen_capture = ScreenCapture.objects.all().order_by('-taken_at').first()
            if screen_capture != None:
                similarity = screen_capture.compare_image(image)
                if similarity < 3:
                    continue

            image = pillow_to_image_field(image)
            ScreenCapture.objects.create(image=image,input_name=title)
    
    @staticmethod
    def process_all():
        # Get all the unprocessed images
        nonprocessed_images = ScreenCapture.objects.filter(is_proccessed=False)
        for image in nonprocessed_images:
            image.process_image()

    def compare_image(self,image_2:PILImage):
        """
        Compare two images.
        Return a float between 0 and 1.
        """

        # Read image
        image_1 = Image.open(self.image.path)
        hash_1 = imagehash.dhash(image_1)
        hash_2 = imagehash.dhash(image_2)
        return (hash_1 - hash_2)
                
    def process_image(self,skin_threshold=5,parse_images=True):
        """
        Convert a raw screen capture to a processed screen capture.
        """
        # Convert the raw screen capture to a PIL image
        image = Image.open(self.image.path)


        if parse_images:
            images = parse_screenshot_to_real_pictures(image)
        else:
            images = [image]
        
        # Process the images
        skin_percentage_results = []
        nsfw_rating_results = []
        for image in images:

            skin_percentage = get_skin_rating_of_image(image)

            if skin_percentage < skin_threshold:
                continue
            else:
                skin_percentage_results.append(skin_percentage)
            
            nsfw_rating = get_nsfw_rating_of_image(image)
            nsfw_rating_results.append(nsfw_rating)
        
        # Clean up the nsfw_rating_results

        nsfw_rating_results = [i for i in nsfw_rating_results if i > .5]


        # Save the results
        try:
            self.skin_percentage = sum(skin_percentage_results)/len(skin_percentage_results)
        except ZeroDivisionError:
            self.skin_percentage = 0
        try:
            self.average_nsfw = sum(nsfw_rating_results)/len(nsfw_rating_results)
        except ZeroDivisionError:
            self.average_nsfw = 0
        try:
            self.max_nsfw = max(nsfw_rating_results)
        except ValueError:
            self.max_nsfw = 0
        self.is_parsed = parse_images
        self.is_thresholded = skin_threshold > 0
        self.is_proccessed = True
        self.save()

