# alerts/views.py

from django.shortcuts import render
from rest_framework import viewsets
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
    queryset = (
        Alert.objects
        .prefetch_related("transcript_set")
        .order_by("-timestamp")
    )
    serializer_class = AlertSerializer

    # Optional: Override the create method to add logging
    def perform_create(self, serializer):
        serializer.save()
        logger.info(f"[API] Alert created via ViewSet: {serializer.instance}")

# --- UI Views ---


def dashboard(request):
    """Renders the main dashboard page."""
    return render(request, 'dashboard.html')


class TranscriptViewSet(viewsets.ModelViewSet):
    queryset = Transcript.objects.all()
    serializer_class = TranscriptSerializer
