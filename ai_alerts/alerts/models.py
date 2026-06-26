from django.db import models


class Alert(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    # stores list like ["NO_HELMET", "NO_VEST"]
    violations = models.TextField(blank=True, null=True)

    camera_id = models.CharField(max_length=100, blank=True, null=True)
    video = models.FileField(upload_to="", blank=True, null=True)

    def __str__(self):
        return f"Alert {self.id}: {self.violations} at {self.timestamp.strftime('%Y-%m-%d %H:%M')}"


class Transcript(models.Model):
    alert = models.ForeignKey(Alert, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    summary = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"Transcript for Alert {self.alert_id} at {self.timestamp.strftime('%Y-%m-%d %H:%M')}"

