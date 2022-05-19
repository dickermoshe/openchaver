import logging
import traceback
logger = logging.getLogger('django')

from PIL import Image
import cv2 as cv
import numpy as np
from mss import mss
from pywinauto import Desktop
import tensorflow as tf

from pywinauto.win32structures import RECT
from PIL.Image import Image as PILImage
from pywinauto.controls.uiawrapper import UIAWrapper

from django.conf import settings

interpreter = tf.lite.Interpreter(model_path=settings.AI_MODEL_PATH)
interpreter.allocate_tensors()
sct = mss()
desktop = Desktop(backend="uia")

# Mask out all pixels that are identical to their neighbors
def _mask_out_identical_pixels(img : np.ndarray,roll:int = 1) -> np.ndarray:

    # Shift the image up one pixel
    shifted_up = np.abs(img-np.roll(img, roll, axis=0))
    # Shift the image down one pixel
    shifted_down = np.abs(img-np.roll(img, roll * -1, axis=0))
    # Shift the image left one pixel
    shifted_left = np.abs(img-np.roll(img, roll, axis=1))
    # Shift the image right one pixel
    shifted_right = np.abs(img-np.roll(img, roll * -1, axis=1))
    # Add the shifted images together
    matte = shifted_up + shifted_down + shifted_left + shifted_right
    # Convert to a grayscale image
    matte = cv.cvtColor(matte, cv.COLOR_BGR2GRAY)
    # Threshold the image
    _, matte = cv.threshold(matte, 0, 1, cv.THRESH_BINARY)
    # Make into 3 channels
    matte = cv.merge([matte, matte, matte])
    # Matte out
    img = cv.multiply(img,matte)

    return img

# Get the active window
def get_active_window():
    try:
        desktop_windows = desktop.windows()
    except:
        logger.error('Could not retrieve any windows')
        logger.debug(traceback.format_exc())
        return None
    
    for window in desktop_windows:
        try:
            is_active = window.is_active()
            window_title = window.window_text()
            logger.debug(f'Window title: {window_title}')
            logger.debug(f'Active: {is_active}')

            if is_active and window_title not in [
                'Program Manager',
                'Taskbar',]:
                logger.debug(f'Active window: {window_title}')
                return window
            
        except:
            logger.error('Could not retrieve window title/active status. Skipping.')
            logger.debug(traceback.format_exc())
            continue
    return None

# The first function return the coordinates of the picture we want to take
def get_coordinates_on_screen(source:str) -> list[dict[str,int]]:
    
    def _rectangle_to_coordinates(rectangle:RECT) -> dict[str,int]:
        top = rectangle.top
        left = rectangle.left
        width = rectangle.width()
        height = rectangle.height()
        return {'top':top,'left':left,'width':width,'height':height}

    if isinstance(source,UIAWrapper):
        logger.debug('Getting coordinates from UIAWrapper')

        try:
            rect = source.rectangle()
            coordinates = _rectangle_to_coordinates(rect)
            return [coordinates]
        except:
            logger.error('Could not get rectangle from UIAWrapper')
            logger.debug(traceback.format_exc())
            return None

    elif source == 'monitor':

        logger.debug('Getting coordinates from monitor')
        logger.debug(f'Monitors: {sct.monitors}')

        if len(sct.monitors) == 1:
            return sct.monitors
        elif len(sct.monitors) > 1:
            return sct.monitors[1:]
        else:
            return None
    
# Adjust the size of the coordinates to fit the screen
def fit_coordinates_to_monitor(coordinates:dict[str,int]) -> dict[str,int]:
    # Get the monitor
    monitor = sct.monitors[0]

    # Adjust the size of the monitor coordinates to remove the taskbar
    adjusted_monitor_height = monitor['height'] - 40 if monitor['height'] > 40 else monitor['height']

    # Ensure that top and left are positive
    top = max(0, coordinates['top'])
    left = max(0, coordinates['left'])
    

    # Ensure top and left are within the monitor
    top = min(adjusted_monitor_height, top)
    left = min(monitor['width'], left)

    # Ensure that the width and height are within the monitor
    width = min(coordinates['width'] , monitor['width'])
    height = min(coordinates['height'] ,adjusted_monitor_height)

    return {'top':top,'left':left,'width':width,'height':height}

# Take a screenshot of the coordinates
def take_picture_of_coordinates(coordinates:dict[str,int]) -> PILImage:
    # Take a screenshot of the coordinates
    try:
        screenshot = sct.grab(coordinates)
    except:
        logger.error('Could not take screenshot')
        logger.debug(traceback.format_exc())
        return None
    try:
        if screenshot.size.width > 0 and screenshot.size.height > 0:
            return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        else:
            return None
    except:
        logger.error('Could not convert screenshot to PILImage')
        logger.debug(traceback.format_exc())
        return None

# Get Title of the active window
def get_title_of_window(window) -> str:
    try:
        return window.window_text()
    except:
        logger.error('Could not get title of window. Returning blank string')
        logger.debug(traceback.format_exc())
        return ''

# Get the skin rating of an image
def get_skin_rating_of_image(img : PILImage) -> float:
    
    
    # Pillow to OpenCV and change to BGR
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    
    # Create a new image that is completely black besides any pixels that are the same between the two images
    #img = _mask_out_identical_pixels(img,roll=5)
    img = _mask_out_identical_pixels(img,roll=2)
    img = _mask_out_identical_pixels(img,roll=1)

    #converting from gbr to hsv color space
    img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    #skin color range for hsv color space 
    HSV_mask = cv.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv.morphologyEx(HSV_mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    #skin color range for hsv color space 
    YCrCb_mask = cv.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv.morphologyEx(YCrCb_mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv.medianBlur(global_mask,3)
    global_mask = cv.morphologyEx(global_mask, cv.MORPH_OPEN, np.ones((4,4), np.uint8))
    global_result=cv.bitwise_not(global_mask)

    # Get a percentage of skin pixels
    total_pixels = global_result.shape[0] * global_result.shape[1]
    skin_pixels = cv.countNonZero(cv.bitwise_not(global_result))
    skin_percentage = (skin_pixels / total_pixels) * 100
    return skin_percentage

# Get the nsfw rating of an image
def get_nsfw_rating_of_image(img : PILImage) -> float:
    IMAGE_DIM = 256
    img = np.array(img)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img = np.asarray(img,dtype='float32')
    if len(img.shape) == 3:
        pass
    elif len(img.shape) == 2:
        img = img.reshape((img.shape[0], img.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (img.shape,))
    img /= 255
    img = np.asarray([img])[0]
    img = cv.resize(img, (IMAGE_DIM, IMAGE_DIM))
    img = np.expand_dims(img, axis=0)
    input_data = np.array(img, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return  output_data[0][0]

# Parse screenshot into a list of images
def parse_screenshot_to_real_pictures(img:PILImage) -> list[PILImage]:

    # Convert the image to OpenCV format
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

    # Create a new image that is completely black besides any pixels that are the same between the two images
    shifted_up = np.roll(img, 1, axis=0)
    # Shift the image down one pixel
    shifted_down = np.roll(img, -1, axis=0)
    # Shift the image left one pixel
    shifted_left = np.roll(img, 1, axis=1)
    # Shift the image right one pixel
    shifted_right = np.roll(img, -1, axis=1)

    # Blur the images together
    t_img = cv.addWeighted(shifted_up, 1, shifted_down, 0, 0.0)
    t_img = cv.addWeighted(t_img, 1, shifted_left, 0, 0.0)
    mask = cv.addWeighted(t_img, 1, shifted_right, 0, 0.0)

    # Create a new image that is completely black besides any pixels that are the same between the two images
    diff = cv.absdiff(img, mask)

    # Threshhold any pixels that are not black
    _, thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY)

    # Make the image grayscale
    gray = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)

    # Blur the image
    #blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Find the contours
    contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Remove the contours that are too small
    contours = [c for c in contours if cv.contourArea(c) > 1000]


    bounding_boxes = []

    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        # if box is too short or thin, skip it
        if w < 100 or h < 100:
            continue

        bounding_boxes.append((x,y,w,h))
    
    # Save the image
    images= []
    for box in bounding_boxes:
        x,y,w,h = box
        cropped_image = img[y:y+h, x:x+w]
        if cropped_image.size != 0:
            images.append(cropped_image)
    
    # Convert the images to PIL format
    images = [Image.fromarray( cv.cvtColor(img , cv.COLOR_RGB2BGR) ) for img in images]
    
    return images

