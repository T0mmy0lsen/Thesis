import json

from django.core import serializers
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.urls import path

from explore.process.main import Main


def index(request):
    result = serializers.serialize('json', Main().run())
    return JsonResponse({'data': json.loads(result)})

