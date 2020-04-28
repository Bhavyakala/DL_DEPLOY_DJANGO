from django.db import models
from datetime import datetime

# change names away from tutorial_... 
class TutorialCategory(models.Model):
    tutorial_category = models.CharField(max_length=100)
    category_summary = models.CharField(max_length=200)
    category_slug = models.CharField(max_length=200)

    class Meta:
        # Django doesn't handle plurals well
        verbose_name_plural = "Categories"

    def __str__(self):
        return self.tutorial_category


class TutorialSeries(models.Model):
    tutorial_series = models.CharField(max_length=200)
    tutorial_category = models.ForeignKey(TutorialCategory, default=1, verbose_name="Category", on_delete=models.SET_DEFAULT)
    # on_delete will set the categories of a deleted category to default
    
    series_summary = models.CharField(max_length=300)
    
    class Meta:
        # Django doesn't handle plurals well
        verbose_name_plural = "Series"

    def __str__(self):
        return self.tutorial_series



class Tutorial(models.Model):
    tutorial_title = models.CharField(max_length=200)
    tutorial_content = models.TextField()
    tutorial_published = models.DateTimeField("date published", default=datetime.now)

    tutorial_series = models.ForeignKey(TutorialSeries, default=1,verbose_name="Series", on_delete=models.SET_DEFAULT )
    tutorial_slug = models.CharField(max_length=200, default=1)

    # converts objects to strings so they can be passed more easily to HTML
    def __str__(self):
        return self.tutorial_title

class Inference(models.Model) :
    
    image = models.ImageField(null=True)
    prediction = models.CharField(max_length=200)
