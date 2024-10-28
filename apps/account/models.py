from django.contrib.auth.models import AbstractUser
from django.db import models
import uuid


class CustomUser(AbstractUser):
    # Добавляем поле для телефона
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    phone = models.CharField(max_length=15, unique=True, blank=True, null=True)

    def __str__(self):
        return self.username
