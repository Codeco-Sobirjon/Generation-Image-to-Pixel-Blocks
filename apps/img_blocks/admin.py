from django.contrib import admin

from apps.img_blocks.models import ImageModel, ImageSchemas, SaveAsPDF


class ImageModelAdmin(admin.ModelAdmin):
    list_display = ['uuid', 'parent']
    ordering = ['-created_at']


class ImageSchemasAdmin(admin.ModelAdmin):
    list_display = ['uuid', 'author', 'created_at']
    ordering = ['-created_at']


class FileAsPDFAdmin(admin.ModelAdmin):
    list_display = ['uuid', 'author', 'created_at']


admin.site.register(ImageModel, ImageModelAdmin)
admin.site.register(ImageSchemas, ImageSchemasAdmin)
admin.site.register(SaveAsPDF, FileAsPDFAdmin)
