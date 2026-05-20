# alerts/views.py

from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.views import APIView
from .serializers import AlertSerializer, TranscriptSerializer
from .models import Alert, Transcript
import logging

# Get an instance of a logger for better debugging
logger = logging.getLogger(__name__)

# --- Primary API View ---


class AlertViewSet(viewsets.ModelViewSet):
    """
    This is now the single source of truth for all API logic.
    It handles creating, listing, retrieving, updating, and deleting alerts.
    """
    queryset = Alert.objects.all().order_by('-timestamp')
    serializer_class = AlertSerializer

    # Optional: Override the create method to add logging
    def perform_create(self, serializer):
        serializer.save()
        logger.info(f"[API] Alert created via ViewSet: {serializer.instance}")

# --- Legacy/Proxy View ---


class CreateAlertView(APIView):
    """
    This view now acts as a "proxy" or "alias" to the AlertViewSet.
    Any requests to its URL will be safely forwarded, ensuring nothing breaks.
    """

    def post(self, request, *args, **kwargs):
        # This line forwards the request to the 'create' action of the AlertViewSet
        create_action = AlertViewSet.as_view({'post': 'create'})
        logger.warning(
            f"[API] Request received on legacy CreateAlertView endpoint. Forwarding to AlertViewSet.")
        return create_action(request, *args, **kwargs)

# --- UI Views (Unchanged) ---


def alerts_ui(request):
    """Renders a UI page to show the latest alerts."""
    alerts = Alert.objects.order_by('-timestamp')[:50]
    return render(request, "alerts_ui.html", {"alerts": alerts})


def dashboard(request):
    """Renders the main dashboard page."""
    return render(request, 'dashboard.html')


class TranscriptViewSet(viewsets.ModelViewSet):
    queryset = Transcript.objects.all()
    serializer_class = TranscriptSerializer
