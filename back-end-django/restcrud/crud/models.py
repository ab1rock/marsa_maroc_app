from django.db import models

def upload_path(instance,filename):
    return '/'.join(['uploads'],filename)


class Container(models.Model):
    code = models.CharField(max_length=255,default="AJSJ")
    date_time = models.DateTimeField()
    image_input = models.ImageField(upload_to=upload_path)  # Image d'entrée
    image_output = models.ImageField(upload_to=upload_path)  # Image de sortie (après traitement)
    detection_threshold = models.FloatField(default=0.9) 
    

    def __str__(self):
        return self.code

