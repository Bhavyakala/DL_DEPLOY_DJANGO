from django import forms
from django.forms import ModelForm

class UploadForm(forms.Form) :
    picture = forms.ImageField()