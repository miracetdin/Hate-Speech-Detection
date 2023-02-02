from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="home"),
    path("index", views.index, name="home2"),
    path("dataset", views.dataset, name="dataset"),
    path("example", views.examples, name="examples")
]
