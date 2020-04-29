from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .models import  Inference
from .forms import UploadForm
from rest_framework.parsers import JSONParser
from . import inference
from django.contrib import messages
import h5py
import os
base_dir = os.path.dirname(os.path.realpath(__file__))

# Create your views here.
def upload_image(request) :

    if request.method == "POST":
        uploadForm = UploadForm(request.POST, request.FILES)
        if uploadForm.is_valid() :
            inf = Inference()
            inf.image =  uploadForm.cleaned_data['picture']
            image = request.FILES['picture']
            inf.caption = get_prediction(image)
            inf.save()
            messages.info(request,f"upload success")
            return render(request = request, template_name="main/caption-generator.html", context={"form" : uploadForm , 
                                                                                                   "caption" : inf.caption,
                                                                                                   "image" : inf.image})            
        else :
            messages.info(request,f"upload not success")
            return render(request = request, template_name="main/caption-generator.html", context={"form" : uploadForm})

    uploadForm = UploadForm()
    return render(request = request, template_name="main/caption-generator.html", context={"form" : uploadForm})        

def get_prediction(image) :
    return inference.run_inference(os.path.join(base_dir,"ml_models/model-ep001-loss3.283-val_loss3.734.h5"), 2, image)