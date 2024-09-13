from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Container
from .serializers import ContainerSerializer
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.conf import settings
import os


@api_view(['GET', 'PUT', 'DELETE'])
def container_detail(request, pk):
    try:
        container = Container.objects.get(pk=pk)
    except Container.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = ContainerSerializer(container)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = ContainerSerializer(container, data=request.data,partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        container.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


###################################################################################
@api_view(['GET', 'POST'])
def container_list(request):
    if request.method == 'GET':
        containers = Container.objects.all()
        serializer = ContainerSerializer(containers, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = ContainerSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#####################################################################################
### image processing : 
import os
import cv2
import supervision as sv
from ultralytics import YOLO
from django.http import JsonResponse
import uuid
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .models import Container 
import easyocr
import torch

@csrf_exempt
def upload_and_process_image(request):
    if request.method == 'POST':
        file = request.FILES.get('image')
        if not file:
            return JsonResponse({'error': 'No file provided'}, status=400)

        # Définir le chemin pour enregistrer le fichier temporairement
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        # Générer un UUID unique pour cette session de traitement
        session_uuid =str(uuid.uuid4())[:4]

        # Générer un nom de fichier unique pour l'image téléchargée
        unique_filename = f"{session_uuid}{os.path.splitext(file.name)[1]}"
        file_path = os.path.join(upload_dir, unique_filename)

        # Sauvegarder le fichier uploadé
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Charger le modèle YOLO
        try:
            model = YOLO('../pre_trained_on_roboflow_dataset.pt')  # Remplacez par le chemin de votre modèle YOLO
            model.to('cuda')  # Charger le modèle sur le GPU
        except AttributeError as e:
            return JsonResponse({'error': f"Error loading model: {str(e)}"}, status=500)

        # Charger l'image téléchargée
        image = cv2.imread(file_path)

        if image is None:
            return JsonResponse({'error': f"Error: Unable to load image at {file_path}"}, status=500)

        # Redimensionner l'image
        resized_image = cv2.resize(image, (640, 480))

        # Appliquer la détection avec YOLO
        try:
            results = model(resized_image, device='cuda')[0]  # Utiliser CUDA pour l'inférence
        except Exception as e:
            return JsonResponse({'error': f"Error during model prediction: {str(e)}"}, status=500)

        # Convertir les résultats YOLO en détections
        detections = sv.Detections.from_ultralytics(results)

        # Enregistrer les images recadrées basées sur les boîtes englobantes
        margin = 6
        cropped_images = []
       
        for i, box in enumerate(detections.xyxy[:1]):
            x1, y1, x2, y2 = map(int, box[:4])
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(resized_image.shape[1], x2 + margin)
            y2 = min(resized_image.shape[0], y2 + margin)

            cropped_image = resized_image[y1:y2, x1:x2]
            cropped_image_filename = f"cropped_image_{session_uuid}_{i}.jpg"
            cropped_image_path = os.path.join(upload_dir, cropped_image_filename)
            cv2.imwrite(cropped_image_path, cropped_image)
            cropped_images.append(cropped_image_path)
   
        # Annoter l'image originale
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=resized_image, detections=detections)
        annotated_image_filename = f"annotated_image_{session_uuid}.jpg"
        annotated_image_path = os.path.join(upload_dir, annotated_image_filename)
        cv2.imwrite(annotated_image_path, annotated_image)
        
        date_timee = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
        
        ####### OCR ZONE ########
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

       
        # Replace with the path to your image
        img = cv2.imread(cropped_image_path)

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply EasyOCR
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())  # Load the model for English with GPU if available
        result = reader.readtext(img_gray)
        
        # Combine all detected text into a single line and calculate the average probability
        combined_text = ' '.join([text for _, text, _ in result])
        average_probability = round(sum([prob for _, _, prob in result]) / len(result), 2) if result else 0


        truncated_text = combined_text[:11]

        fixed_code = list(truncated_text)

        # Iterate through each character starting from the 5th character (index 4)
        for i in range(4,len(fixed_code)):
            if fixed_code[i] == 'I':
                fixed_code[i] = '1'
            elif fixed_code[i] == 'A':
                fixed_code[i] = '4'
            elif fixed_code[i] == 'G':
                fixed_code[i] = '6'
            elif fixed_code[i] == 'B':
                fixed_code[i] = '8'
            elif fixed_code[i] == 'J':
                fixed_code[i] = '9'
            elif not (fixed_code[i].isdigit() or fixed_code[i].isalpha()):
                fixed_code[i] = ''
            

        # Join the fixed_code list back into a string
        corrected_code = ''.join(fixed_code)
        corrected_code=corrected_code[:11]
        cv2.waitKey(0)
        cv2.destroyAllWindows()
       
        container = Container.objects.create(
            code=corrected_code,
            image_input=f"uploads/{annotated_image_filename}",
            image_output=f"uploads/{os.path.basename(cropped_images[0])}" if cropped_images else None,
            detection_threshold=average_probability, # Exemple de valeur
            date_time=date_timee
        ),
        

        return JsonResponse({
            'message': 'Image processed successfully',
            'code':corrected_code,
            'input': os.path.basename(file_path),
            'image_input': os.path.basename(annotated_image_path),
            'image_output': [os.path.basename(img) for img in cropped_images],
            'detection_threshold':average_probability,
            'date_time': date_timee
        }, status=201)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

#################################################################################################################


####pour la video 

import cv2
import os
import supervision as sv
from ultralytics import YOLO
import torch
import easyocr
from django.utils import timezone
from django.http import StreamingHttpResponse, JsonResponse
import uuid
from django.conf import settings
from .models import Container

def process_video_stream():
    video_url = "http://192.168.11.6:8080/video"  # URL de votre flux vidéo
    cap = cv2.VideoCapture(video_url)
    
    if not cap.isOpened():
        raise Exception('Unable to open video stream')
    frame_skip = 5  # Process every 5th frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        annotated_frame, container_data = process_frame(frame)
        
        # Encoder l'image annotée pour l'envoi en streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

################################

def process_frame(frame):
    # Charger le modèle YOLO
    try:
        model = YOLO('../pre_trained_on_roboflow_dataset.pt')  # Replace with the path to your YOLO model
        model.to('cuda')  # Move the model to GPU
    except AttributeError as e:
        print("Error loading model:", e)
    
    # Redimensionner la frame
    resized_frame = cv2.resize(frame, (640, 480))
    
    # Appliquer la détection avec YOLO
    results = model(resized_frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    
    # Générer un UUID unique pour cette session de traitement
    session_uuid = str(uuid.uuid4())[:4]
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    # Sauvegarder les images annotées et recadrées
    margin = 6
    cropped_images = []
    cropped_image_path = None
    
    if len(detections.xyxy) > 0:
        for i, box in enumerate(detections.xyxy[:1]):
            x1, y1, x2, y2 = map(int, box[:4])
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(resized_frame.shape[1], x2 + margin)
            y2 = min(resized_frame.shape[0], y2 + margin)

            cropped_image = resized_frame[y1:y2, x1:x2]
            cropped_image_filename = f"cropped_image_{session_uuid}_{i}.jpg"
            cropped_image_path = os.path.join(upload_dir, cropped_image_filename)
            cv2.imwrite(cropped_image_path, cropped_image)
            cropped_images.append(cropped_image_path)
            
    annotated_frame = sv.BoxAnnotator().annotate(scene=resized_frame, detections=detections)
    annotated_image_filename = f"annotated_image_{session_uuid}.jpg"
    annotated_image_path = os.path.join(upload_dir, annotated_image_filename)
    cv2.imwrite(annotated_image_path, annotated_frame)
    
    ####### OCR ZONE ########
    
    if cropped_images:  # Vérifie s'il y a des images recadrées
        cropped_image_path = cropped_images[0]
    else:
        print("No cropped images found, skipping OCR.")
        return annotated_frame, None
    
    img = cv2.imread(cropped_image_path)
    if img is None:
        raise ValueError(f"Failed to read image at: {cropped_image_path}")
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    result = reader.readtext(img_gray)
    combined_text = ' '.join([text for _, text, _ in result])
    average_probability = round(sum([prob for _, _, prob in result]) / len(result), 2) if result else 0

    truncated_text = combined_text[:11]
    fixed_code = list(truncated_text)

    for i in range(4, len(fixed_code)):
        if fixed_code[i] == 'I':
            fixed_code[i] = '1'
        elif fixed_code[i] == 'A':
            fixed_code[i] = '4'
        elif fixed_code[i] == 'G':
            fixed_code[i] = '6'
        elif fixed_code[i] == 'B':
            fixed_code[i] = '8'
        elif fixed_code[i] == 'J':
            fixed_code[i] = '9'
        elif not (fixed_code[i].isdigit() or fixed_code[i].isalpha()):
            fixed_code[i] = ''
    
    corrected_code = ''.join(fixed_code)[:11]

    # Sauvegarder dans la base de données
    date_timee = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
    container = Container.objects.create(
        code=corrected_code,
        image_input=f"uploads/{annotated_image_filename}",
        image_output=f"uploads/{os.path.basename(cropped_images[0])}" if cropped_images else None,
        detection_threshold=average_probability,
        date_time=date_timee
    )

    return annotated_frame, container








def stream_video(request):
    return StreamingHttpResponse(process_video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')



#################


from django.http import JsonResponse
from .models import Container

from django.utils.timezone import now

def latest_processed_data(request):
    try:
        today = now().date()
        container = Container.objects.filter(date_time__date=today).latest('date_time')
        response_data = {
            'message': 'Frame processed successfully',
            'code': container.code,
            'input': container.image_input.url,
            'image_output': container.image_output.url,
            'detection_threshold': container.detection_threshold,
            'date_time': container.date_time
        }
        return JsonResponse(response_data, status=200)
    except Container.DoesNotExist:
        return JsonResponse({}, status=200)














#############################################################################################
 # views.py
import cv2
import supervision as sv
from ultralytics import YOLO
import torch
from django.http import StreamingHttpResponse

# URL of your video stream
video_url = "http://192.168.1.6:8080/video"

# Load the YOLO model
try:
    model = YOLO('../pre_trained_on_roboflow_dataset.pt')  # Replace with the path to your YOLO model
    model.to('cuda')  # Move the model to GPU
except AttributeError as e:
    print("Error loading model:", e)

# Initialize annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def generate_frames():
    # Open the video stream
    cap = cv2.VideoCapture(video_url)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to the expected input size
            resized_frame = cv2.resize(frame, (640, 640))  # YOLO model typically expects 640x640

            # Convert the frame from NumPy to a PyTorch tensor
            frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float().to('cuda') / 255.0

            # Process frame for detection
            try:
                results = model(frame_tensor)[0]
            except Exception as e:
                print("Error during model prediction:", e)
                continue

            detections = sv.Detections.from_ultralytics(results)
            annotated_image = box_annotator.annotate(scene=resized_frame, detections=detections)
            
            # Encode the image to JPEG format (annotated_image is already a NumPy array)
            _, buffer = cv2.imencode('.jpg', annotated_image)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()  # Release the video capture object
        cv2.destroyAllWindows()  # Close all OpenCV windows

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

