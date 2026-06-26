# alerts/serializers.py

from rest_framework import serializers
from .models import Alert, Transcript


class AlertSerializer(serializers.ModelSerializer):
    # This new field will contain the full URL for the video
    video_url = serializers.SerializerMethodField()
    summary = serializers.SerializerMethodField()

    class Meta:
        model = Alert
        # We list all fields and add our new 'video_url' and 'summary' fields
        fields = ['id', 'timestamp', 'violations',
                  'camera_id', 'video', 'video_url', 'summary']

    def get_video_url(self, obj):
        """
        This method creates the full URL for the video file.
        """
        if obj.video:
            return obj.video.url
        return None

    def get_summary(self, obj):
        """
        Retrieves the associated transcript summary for this alert.
        """
        transcript = obj.transcript_set.first()
        return transcript.summary if transcript else "No summary available."



class TranscriptSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transcript
        fields = '__all__'
