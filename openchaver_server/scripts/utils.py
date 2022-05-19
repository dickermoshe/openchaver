import psutil
from django.conf import settings

def get_current_cpu_usage():
    """
    Get the current CPU usage.
    """
    return psutil.cpu_percent()

def get_size_of_screenshot_folder():
    """
    Get the size of the screenshot folder.
    """
    path = settings.MEDIA_ROOT / 'screenCaptures'
    return sum(file.stat().st_size for file in path.rglob('*'))