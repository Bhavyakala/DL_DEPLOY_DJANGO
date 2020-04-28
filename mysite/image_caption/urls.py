from . import views
from django.urls import path

app_name = "image_caption"

urlpatterns = [
    path('', views.upload_image, name='image-caption-generator'),
    
]
