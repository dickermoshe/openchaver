from pprint import pprint as pp
from typing import TypedDict

from PIL.Image import Image as PILImage
from PIL import Image

from mss import mss
from mss.screenshot import ScreenShot

import numpy as np
import cv2 as cv

from .windows import Apps

class CameraRoll(TypedDict):
    """
    A film is a list of images
    """
    images: list[PILImage]
    title: str

class Camera:
    def __init__(self) -> None:
        self.sct = mss()
    
    def _pillow_to_opencv(self,image : PILImage) -> np.ndarray:
        """
        Converts the image to the specified format.
        :param image: The image to convert.
        """
        return cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    
    def _opencv_to_pillow(self,image : np.ndarray) -> PILImage:
        """
        Converts the image to the specified format.
        :param image: The image to convert.
        """
        return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    def _mss_to_opencv(self,screenshot:ScreenShot) -> np.ndarray:
        """
        Converts the image to the specified format.
        :param image: The image to convert.
        """
        return np.array(screenshot)
    
    def _mss_to_pillow(self,screenshot:ScreenShot) -> PILImage:
        """
        Converts the image to the specified format.
        :param image: The image to convert.
        """
        return self._opencv_to_pillow(self._mss_to_opencv(screenshot))

    def take_screenshot_of_monitor(self,monitor : int = 0) -> CameraRoll:
        """
        Takes a screenshot of the monitor.
        :param monitor: The monitor to take a screenshot of. 0 will take of all the monitors.
        :returns A dictionary containing the title and the images.
        """
        # Create a list of all the screens
        if monitor == 0:
            monitor_ids = range(1,len(self.sct.monitors))
        elif monitor < len(self.sct.monitors):
            monitor_ids = [monitor]
        else:
            raise ValueError("The monitor id is out of range.")
        
        # Take a screenshot of each monitor
        images = []
        for monitor_id in monitor_ids:
            monitor = self.sct.monitors[monitor_id]
            sct_img = self.sct.grab(monitor)
            images.append(self._mss_to_pillow(sct_img))

        title = 'Monitor'

        # Return the images
        return {'title':title,'images':images}

    def take_screenshot_of_active_window(self) -> CameraRoll:
        """
        Takes a screenshot of the active window.
        :returns A dictionary containing the title and the images.
        """
        # Get the list of all the windows
        apps = Apps()
        active_window = apps.get_active_window()
        
        images = [active_window.capture_as_image()]
        title = active_window.window_text()

        # Return the images
        return {'title':title,'images':images}
    

    