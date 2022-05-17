from typing import Dict
from .camera import Camera
from .detector import Detector
from .parser import Parser
import cv2  as cv
from PIL.Image import Image as PILImage
from PIL import Image

class Checker:
    def __init__(self) -> None:
        self.camera = Camera()
        self.detector = Detector()

    def check_image(self,image:PILImage, parse_images:bool=True,skin_threshold:float=0) -> list[PILImage]:
        """
        Checks the specified input for inappropriate content.

        :param image: The image to check.
        :param parse_images: Whether to parse individual images from the input or to process the entire input at once.
        :param skin_threshold: How much skin content needed to run the nsfw detection. If you want to run the nsfw detection even if there is no skin content, set this to 0.
        """

        if parse_images:
            images = Parser.parse_real_pictures(self.camera._pillow_to_opencv(image))
        else:
            images = [image]
        
        results = []

        for img in images:
            # Get the skin rating
            skin_rating = self.detector.skin_rating_of_image(img)

            if skin_rating['percentage'] < skin_threshold:
                continue
            else:
                nsfw_rating = self.detector.nsfw_rating_of_image(img)
                results.append({'image':img,'skin_rating':skin_rating,'nsfw_rating':nsfw_rating})
            
        return results




        


    
                

        

        







    


    
        
    


        


