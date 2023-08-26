import json
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import ImageSerializer  # Update with the correct path
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from PIL import Image

preprocess_input = tf.keras.applications.resnet50.preprocess_input

def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # Applying preprocessing for ResNet50
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

class PredictionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        serializer = ImageSerializer(data=request.data)
        
        if serializer.is_valid():
            uploaded_image = serializer.validated_data['image']
            uploaded_image = Image.open(uploaded_image)
            uploaded_image = uploaded_image.resize((150,150))

            tf.keras.preprocessing.image.img_to_array(uploaded_image)
            model_path = os.path.abspath("/Coding/Django/Project/Acne/Model")
            print("Absolute model path:", model_path)

            model = tf.saved_model.load(model_path)

            input_image = np.expand_dims(uploaded_image, axis=0)  
            input_image = tf.keras.applications.mobilenet.preprocess_input(input_image)  

            infer = model.signatures["serving_default"]

            print("Input keys:", infer.structured_input_signature)
            print("Output keys:", infer.structured_outputs)

            predictions = infer(tf.constant(input_image))['dense_1']
            array = np.asarray(predictions)
            predictions_list = array.tolist()
            
            formatted_predictions = {
            'acne': predictions_list[0][0],
            'clear': predictions_list[0][1],
            'comedo': predictions_list[0][2]
            }

            closest_class = max(formatted_predictions, key=formatted_predictions.get)
            response_data = {
                'result': closest_class,
                'probability': formatted_predictions[closest_class]
            }

            return Response(response_data)
        else:
            return Response(serializer.errors, status=400)
