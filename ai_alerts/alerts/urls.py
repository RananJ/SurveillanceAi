
from django.urls import path, include
from .views import CreateAlertView
from . import views
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'alerts', views.AlertViewSet, basename='alert')
router.register(r'transcripts', views.TranscriptViewSet, basename='transcript')

ui_patterns = [
    path('ui/', views.alerts_ui, name='alerts_ui'),
    path('create/', CreateAlertView.as_view(), name='create-alert'),
    path('', views.dashboard, name='dashboard'),
]

urlpatterns = [
    path('api/', include(router.urls)),
    *ui_patterns,
]
