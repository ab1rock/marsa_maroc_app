from rest_framework import serializers
from .models import Container

class ContainerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Container
        fields = ['id', 'code', 'date_time','image_input','image_output','detection_threshold']
        extra_kwargs = {
            'date_time': {'required': False},
            'image_input': {'required': False},
            'image_output': {'required': False},
        }

    def update(self, instance, validated_data):
        # Ne mettre à jour que les champs fournis
        instance.code = validated_data.get('code', instance.code)
        instance.detection_threshold = validated_data.get('detection_threshold', instance.detection_threshold)
        
        # Ne pas mettre à jour les images si elles ne sont pas présentes dans les données validées
        instance.save()
        return instance

