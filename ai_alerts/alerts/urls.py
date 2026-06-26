
from django.urls import path, include
from . import views
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'alerts', views.AlertViewSet, basename='alert')
router.register(r'transcripts', views.TranscriptViewSet, basename='transcript')

ui_patterns = [
    path('', views.dashboard, name='dashboard'),
]

urlpatterns = [
    path('api/', include(router.urls)),
    *ui_patterns,
]
