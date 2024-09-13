from django.contrib import admin
from django.urls import path,include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('crud.urls')),  # Inclure les URLs de votre application
]

urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)