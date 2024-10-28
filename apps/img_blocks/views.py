import concurrent.futures
import cv2
import os
import uuid
from io import BytesIO

import numpy as np
from PIL import Image
from django.core.files.base import ContentFile
from django.shortcuts import get_object_or_404
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from sklearn.cluster import KMeans

from .models import ImageModel, ImageSchemas, SaveAsPDF
from .serializers import ImageModelSerializer, ImageListSerializer, SaveAsPdfListSerialzier, SchemasListSerializers, \
    ImagePixelChangeSerializer
from .utils import cut_image_and_save_colors_as_json


class ImageUploadAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        request_body=ImageModelSerializer,
        operation_description="Create a new generate block image",
        tags=['Generate image'],
        responses={201: ImageModelSerializer(many=False)}
    )
    @method_decorator(csrf_exempt)
    def post(self, request, *args, **kwargs):
        user_identifier = None
        response = None
        if not request.user.is_authenticated:
            user_identifier = request.headers.get('user-identifier')
            if not user_identifier:
                user_identifier = str(uuid.uuid4())
                response = Response()
                response.set_cookie('user_identifier', user_identifier)
            else:
                response = None

        data = request.data.copy()
        context_user_identifier = user_identifier or request.user.uuid

        serializer = ImageModelSerializer(data=data, context={'request': request,
                                                              'user_identifier': context_user_identifier})

        if serializer.is_valid():
            serializer.save()
            if response:
                response.data = serializer.data
                response.status_code = status.HTTP_201_CREATED
                return response
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(
        operation_description="Get user's images",
        tags=['Retrieve image'],
        responses={200: ImageListSerializer(many=True)}
    )
    def get(self, request, image_id, *args, **kwargs):
        user_identifier = None  # Initialize user_identifier

        if not request.user.is_authenticated:
            user_identifier = request.headers.get('user-identifier')
            if not user_identifier:
                return Response({'detail': 'User identifier not found'}, status=status.HTTP_400_BAD_REQUEST)

        context_user_identifier = user_identifier or request.user.uuid

        try:
            images = ImageModel.objects.filter(uuid=image_id, user_identifier=context_user_identifier)
            serializer = ImageListSerializer(images, many=True, context={'request': request})
            return Response(serializer.data, status=status.HTTP_200_OK)
        except ImageModel.DoesNotExist:
            return Response({'detail': 'No images found'}, status=status.HTTP_404_NOT_FOUND)


class UpdateImageColors(APIView):
    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter(
                'limit_colors',
                openapi.IN_QUERY,
                description="Limit for the number of colors",
                type=openapi.TYPE_INTEGER
            ),
        ],
        tags=['Generate image'],
    )
    def get(self, request, image_id):
        user_identifier = None  # Initialize user_identifier

        if not request.user.is_authenticated:
            user_identifier = request.headers.get('user-identifier')
            if not user_identifier:
                return Response({'detail': 'User identifier not found'}, status=status.HTTP_400_BAD_REQUEST)

        context_user_identifier = user_identifier or request.user.uuid
        try:
            image_instance = ImageModel.objects.filter(uuid=image_id, user_identifier=context_user_identifier).first()
            limit_colors = request.query_params.get('limit_colors', None)

            if limit_colors is None:
                serializer = ImageListSerializer(image_instance, context={'request': request})
                return Response(serializer.data, status=status.HTTP_200_OK)

            try:
                limit_colors = int(limit_colors)
            except ValueError:
                return Response({'error': 'Invalid limit_colors value. Please provide an integer.'},
                                status=status.HTTP_400_BAD_REQUEST)

            if image_instance.color_image:
                image_path = image_instance.color_image.path
            else:
                image_path = image_instance.main_image.path

            colors = image_instance.main_colors[:limit_colors]

            # Open image and process it
            with Image.open(image_path) as img:
                img_array = np.array(img.convert('RGB'))

            # Convert hex colors to RGB tuples
            rgb_colors = np.array([self.hex_to_rgb(color['hex']) for color in colors], dtype=np.float32)

            # Detect faces and apply selective color quantization
            updated_img_array = self.apply_selective_quantization(img_array, rgb_colors)

            # Save the updated image as a new image
            updated_image = Image.fromarray(updated_img_array.astype(np.uint8))
            new_image_io = BytesIO()
            updated_image.save(new_image_io, format='JPEG')
            new_image_content = ContentFile(new_image_io.getvalue(), name=f"{image_instance.main_image.name}")

            # Create a new ImageModel instance with the updated image
            new_image_instance = ImageModel.objects.create(image=new_image_content, colors=list(colors),
                                                           main_colors=image_instance.main_colors,
                                                           parent=image_instance,
                                                           user_identifier=context_user_identifier,
                                                           main_image=image_instance.main_image,
                                                           color_image=image_instance.color_image)
            new_image_instance.save()

            serializer = ImageListSerializer(new_image_instance, context={'request': request})
            response = Response(serializer.data, status=status.HTTP_200_OK)
            response.set_cookie('user_identifier', context_user_identifier)
            return response

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @staticmethod
    def hex_to_rgb(hex_color):
        return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))

    def apply_selective_quantization(self, img_array, color_list):
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        scale_factor = 0.25
        small_gray_img = cv2.resize(gray_img, None, fx=scale_factor, fy=scale_factor)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(small_gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_mask = np.zeros(img_array.shape[:2], dtype=bool)
        for (x, y, w, h) in faces:
            x, y, w, h = int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)
            face_mask[y:y + h, x:x + w] = True

        non_face_region = np.where(~face_mask)
        face_region = np.where(face_mask)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            non_face_future = executor.submit(self.update_image_colors, img_array[non_face_region], color_list)
            face_future = executor.submit(self.update_image_colors, img_array[face_region], color_list)

            updated_non_face = non_face_future.result()
            updated_face = face_future.result()

            updated_img_array = np.copy(img_array)
            updated_img_array[non_face_region] = updated_non_face
            updated_img_array[face_region] = updated_face

            return updated_img_array

    def update_image_colors(self, img_array, color_list):
        img_array = img_array.reshape((-1, 3))
        distances = np.linalg.norm(img_array[:, None] - color_list, axis=2)
        closest_color_indices = np.argmin(distances, axis=1)
        updated_img_array = color_list[closest_color_indices]
        return updated_img_array.reshape((-1, 3))

    @staticmethod
    def get_biggest_and_smallest_images():
        all_images = ImageModel.objects.all()
        biggest_image = max(all_images, key=lambda img: img.image.width * img.image.height)
        smallest_image = min(all_images, key=lambda img: img.image.width * img.image.height)
        return biggest_image, smallest_image


class UpdateColorsViews(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Update colors in an Image instance",
        responses={200: ImageModelSerializer()},
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'color_id': openapi.Schema(type=openapi.TYPE_INTEGER),
                'new_color_hex': openapi.Schema(type=openapi.TYPE_STRING),
            },
            required=['color_id', 'new_color_hex'],
        ),
    )
    def put(self, request, image_id):
        user_identifier = None  # Initialize user_identifier

        if not request.user.is_authenticated:
            user_identifier = request.headers.get('user-identifier')
            if not user_identifier:
                return Response({'detail': 'User identifier not found'}, status=status.HTTP_400_BAD_REQUEST)

        context_user_identifier = user_identifier or request.user.uuid

        try:
            # Fetch the image instance
            image_instance = ImageModel.objects.filter(uuid=image_id, user_identifier=context_user_identifier).first()
            if not image_instance:
                return Response({'error': 'Image not found'}, status=status.HTTP_404_NOT_FOUND)

            # Get color_id and new_color_hex from the request
            color_id = request.data.get('color_id')
            new_color_hex = request.data.get('new_color_hex')

            # Validate the inputs
            if color_id is None or new_color_hex is None:
                return Response({'error': 'color_id and new_color_hex are required fields'},
                                status=status.HTTP_400_BAD_REQUEST)

            # Find the color in the colors JSON field
            color_instance = next((color for color in image_instance.colors if color['id'] == color_id), None)
            if not color_instance:
                return Response({'error': 'Color not found'}, status=status.HTTP_404_NOT_FOUND)

            # Get the path of the image file
            image_path = image_instance.image.path

            # Call modify_color function
            modify_color(image_path, color_instance['hex'], new_color_hex)

            # Generate a unique filename for the modified image
            modified_image_filename = f'{uuid.uuid4()}.png'

            # Save the modified image back to image_instance.image
            with open('modified_image.png', 'rb') as modified_image_file:
                image_instance.image.save(modified_image_filename, modified_image_file, save=True)
            image_instance.color_image = image_instance.image

            image_instance.save()
            # Optionally, delete the temporary modified image file
            os.remove('modified_image.png')

            # Return success response with updated image instance
            serializer = ImageModelSerializer(image_instance, context={'request': request})
            return Response(serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def modify_color(image_path, old_hex, new_hex, tolerance=10):
    # Open the image
    img = Image.open(image_path)
    img = img.convert("RGBA")  # Convert image to RGBA mode for transparency support
    width, height = img.size

    # Remove '#' from hex strings and convert to RGB tuples
    old_rgb = tuple(int(old_hex.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    new_rgb = tuple(int(new_hex.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

    # Create a blank image for output
    modified_img = Image.new('RGBA', (width, height))

    # Iterate through each pixel in the image
    for x in range(width):
        for y in range(height):
            current_color = img.getpixel((x, y))[:3]  # Get RGB values of current pixel

            # Check if the current pixel color is within tolerance of the old color
            if all(abs(current_color[i] - old_rgb[i]) <= tolerance for i in range(3)):
                # Replace the old color with the new color
                modified_img.putpixel((x, y), new_rgb)
            else:
                # Preserve the original pixel if it doesn't match the old color
                modified_img.putpixel((x, y), img.getpixel((x, y)))

    # Save modified image with a temporary filename
    modified_img.save('modified_image.png')


class BackProcessViews(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Back to process",
        tags=['Back Process'],
        responses={200: ImageListSerializer(many=False)}
    )
    def get(self, request, id):
        images = get_object_or_404(ImageModel, id=id)
        serializer = ImageListSerializer(images, context={'request': request})
        return Response(serializer.data, status=status.HTTP_200_OK)


class GroupedColorsViews(APIView):
    permission_classes = [AllowAny]  # Define your permissions here

    @swagger_auto_schema(
        operation_description="Grouped colors",
        tags=['Grouped colors'],
        responses={200: ImageListSerializer(many=True)}
    )
    def get(self, request, image_id):
        user_identifier = None  # Initialize user_identifier

        if not request.user.is_authenticated:
            user_identifier = request.headers.get('user-identifier')
            if not user_identifier:
                return Response({'detail': 'User identifier not found'}, status=status.HTTP_400_BAD_REQUEST)

        context_user_identifier = user_identifier or request.user.uuid

        try:
            image_instance = ImageModel.objects.filter(uuid=image_id, user_identifier=context_user_identifier).first()
            if not image_instance:
                return Response({'detail': 'Image not found'}, status=status.HTTP_404_NOT_FOUND)

            # Retrieve colors from JSONField
            colors = image_instance.colors

            # Convert hex to RGB
            def hex_to_rgb(hex):
                hex = hex.lstrip('#')
                return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))

            # Extract RGB values from the JSONField 'colors'
            rgb_colors = [hex_to_rgb(color['hex']) for color in colors]

            # Convert to numpy array
            rgb_array = np.array(rgb_colors)

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=10)  # Adjust the number of clusters as needed
            kmeans.fit(rgb_array)
            labels = kmeans.labels_

            # Group colors by cluster labels
            grouped_colors = {}
            for i, label in enumerate(labels):
                if label not in grouped_colors:
                    grouped_colors[label] = []
                grouped_colors[label].append(colors[i])

            # Sort colors within each cluster
            def sort_colors_by_rgb(colors):
                return sorted(colors, key=lambda c: hex_to_rgb(c['hex']))

            sorted_grouped_colors = []
            for cluster_colors in grouped_colors.values():
                sorted_grouped_colors.extend(sort_colors_by_rgb(cluster_colors))

            # Update the image instance with the sorted grouped colors
            image_instance.colors = sorted_grouped_colors
            image_instance.save()

            # Serialize and return the updated image data
            serializer = ImageListSerializer(image_instance, context={'request': request})
            return Response(serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ReturningOwnColorsViews(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Returning own colors",
        tags=['Returning own colors'],
        responses={200: ImageListSerializer(many=True)}
    )
    def get(self, request, image_id):
        user_identifier = None  # Initialize user_identifier

        if not request.user.is_authenticated:
            user_identifier = request.headers.get('user-identifier')
            if not user_identifier:
                return Response({'detail': 'User identifier not found'}, status=status.HTTP_400_BAD_REQUEST)

        context_user_identifier = user_identifier or request.user.uuid

        try:
            image_instance = ImageModel.objects.filter(uuid=image_id, user_identifier=context_user_identifier).first()

            image_instance.colors = image_instance.main_colors[:len(image_instance.colors)]
            image_instance.save()
            serializers = ImageListSerializer(image_instance, context={'request': request})
            return Response(serializers.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MakeSchemasListViews(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Make schemas",
        tags=['Schema'],
        manual_parameters=[
            openapi.Parameter(
                'user-identifier',
                openapi.IN_HEADER,
                description="User identifier",
                type=openapi.TYPE_STRING
            ),
        ],
        responses={200: SchemasListSerializers()}
    )
    def get(self, request, image_id):
        user_identifier = None  # Initialize user_identifier

        if not request.user.is_authenticated:
            user_identifier = request.headers.get('user-identifier')
            if not user_identifier:
                return Response({'detail': 'User identifier not found'}, status=status.HTTP_400_BAD_REQUEST)

        context_user_identifier = user_identifier or request.user.uuid

        try:
            image_instance = ImageModel.objects.filter(uuid=image_id, user_identifier=context_user_identifier).first()

            schema = cut_image_and_save_colors_as_json(image_instance.image.path, 9, 15, )
            queryset = ImageSchemas.objects.create(schema=schema, image=image_instance)
            if not image_instance:
                return Response({'detail': 'Image not found'}, status=status.HTTP_404_NOT_FOUND)

            serializer = SchemasListSerializers(queryset, context={'request': request})
            return Response(serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GetuserSchemasView(APIView):
    permission_classes = [AllowAny, IsAuthenticated]

    @swagger_auto_schema(
        operation_description="Make schemas",
        tags=['Schema'],
        responses={200: SchemasListSerializers()}
    )
    def get(self, request):
        user = request.user
        queryset = ImageSchemas.objects.filter(author=user)
        serializers = SchemasListSerializers(queryset, many=True)
        return Response(serializers.data, status=status.HTTP_200_OK)


class ImagePixelChangeAPIView(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        request_body=ImagePixelChangeSerializer,
        operation_description="",
        tags=['Change image pixels'],
        responses={201: ImagePixelChangeSerializer(many=False)}
    )
    @method_decorator(csrf_exempt)
    def put(self, request, image_id, *args, **kwargs):
        user_identifier = None
        response = None
        if not request.user.is_authenticated:
            user_identifier = request.headers.get('user-identifier')
            if not user_identifier:
                user_identifier = str(uuid.uuid4())
                response = Response()
                response.set_cookie('user_identifier', user_identifier)
            else:
                response = None

        data = request.data.copy()
        context_user_identifier = user_identifier or request.user.uuid

        try:
            image_instance = ImageModel.objects.filter(uuid=image_id, user_identifier=context_user_identifier).first()

            if not image_instance:
                return Response({'detail': 'Image not found'}, status=status.HTTP_404_NOT_FOUND)

            serializer = ImagePixelChangeSerializer(image_instance, data=data, context={'request': request},
                                                    partial=True)

            if serializer.is_valid():
                serializer.save()

                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SaveAsPDFListView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny, IsAuthenticated]

    @swagger_auto_schema(
        request_body=SaveAsPdfListSerialzier,
        operation_description="Create a new generate block image save as file",
        tags=['Save as file'],
        responses={201: SaveAsPdfListSerialzier(many=False)}
    )
    def post(self, request):

        serializer = SaveAsPdfListSerialzier(data=request.data, context={'author': request.user})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(
        operation_description="File Lists",
        tags=['Save as file'],
        responses={200: SaveAsPdfListSerialzier()}
    )
    def get(self, request):
        queryset = SaveAsPDF.objects.filter(author=request.user)
        serializers = SaveAsPdfListSerialzier(queryset, many=True)
        return Response(serializers.data, status=status.HTTP_200_OK)
