from django.db import models

# Create your models here.

class Inference(models.Model) :
    image = models.ImageField(upload_to = 'images/')
    caption = models.CharField(max_length=200,default=None)

