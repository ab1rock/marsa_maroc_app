from django.urls import path
from . import views


urlpatterns = [
    path('containers/', views.container_list),
    path('containers/<int:pk>/', views.container_detail),
    path('upload/',views.upload_and_process_image),
    path('stream_video/', views.stream_video),
    path('latest_data/', views.latest_processed_data), 
]
