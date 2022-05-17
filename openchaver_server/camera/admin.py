from django.contrib import admin
from django.utils.html import format_html

# Register your models here.
from .models import ScreenCapture

@admin.register(ScreenCapture)
class ScreenCaptureAdmin(admin.ModelAdmin):
    
    def image_tag(self, obj):
        return format_html('<img src="{}" width="150"/>'.format(obj.image.url))
    image_tag.short_description = 'Image'

    list_display = ['input_name','is_parsed','is_thresholded','taken_at','average_safe','skin_percentage','image_tag']
    list_filter = ['is_parsed','taken_at','is_thresholded']
    readonly_fields = ('image_tag',)

