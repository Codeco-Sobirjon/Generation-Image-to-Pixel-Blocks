# models.py
from django.db import models
import uuid
from django.utils import timezone

from apps.account.models import CustomUser


class ImageModel(models.Model):
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    image = models.ImageField(upload_to='images/')
    main_image = models.ImageField(upload_to='main_images/')
    color_image = models.ImageField(upload_to='color_images/')
    colors = models.JSONField(null=True, blank=True)
    main_colors = models.JSONField(null=True, blank=True)
    user_identifier = models.CharField(max_length=255)
    parent = models.ForeignKey("self", on_delete=models.CASCADE, null=True, blank=True, default=None)
    pixel_color_codes = models.JSONField(null=True, blank=True)
    color_scheme = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f'Image {self.id}'

    class Meta:
        ordering = ['created_at']


class ImageSchemas(models.Model):
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    author = models.ForeignKey(CustomUser, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ForeignKey(ImageModel, on_delete=models.CASCADE, null=True, blank=True)
    schema = models.JSONField(null=True, blank=True)
    created_at = models.DateField(auto_now_add=True)

    def __str__(self):
        return f'Image {self.id}'

    class Meta:
        ordering = ['created_at']
        

class SaveAsPDF(models.Model):
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    author = models.ForeignKey(CustomUser, on_delete=models.CASCADE, null=True, blank=True)
    file = models.FileField(upload_to='media/files', null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f'Image {self.author}'

    class Meta:
        ordering = ['created_at']
