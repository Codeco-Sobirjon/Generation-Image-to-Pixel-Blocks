from django.urls import path
from apps.img_blocks.views import *

urlpatterns = [
    path('upload/', ImageUploadAPIView.as_view(), name='image-upload'),
    path('images/<uuid:image_id>/', ImageUploadAPIView.as_view(), name='get_user_images'),
    path('color/update/<uuid:image_id>', UpdateColorsViews.as_view()),
    path('update-colors/<uuid:image_id>', UpdateImageColors.as_view(), name='update_image_colors'),
    path('grouped/colors/<uuid:image_id>', GroupedColorsViews.as_view()),
    path('return/own-colors/<uuid:image_id>', ReturningOwnColorsViews.as_view()),

    path('schema/<uuid:image_id>', MakeSchemasListViews.as_view()),
    path('schema', GetuserSchemasView.as_view()),

    path('image/pixel/update/<uuid:image_id>', ImagePixelChangeAPIView.as_view()),
    
    path('file/pdf/', SaveAsPDFListView.as_view())
]
