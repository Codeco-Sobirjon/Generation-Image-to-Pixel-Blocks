from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.permissions import AllowAny
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import get_user_model
from drf_yasg.utils import swagger_auto_schema

from apps.account.serializers import UserRegisterSerializer, UserDetailSerializer, UserLoginSerializer

User = get_user_model()


class RegisterView(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        request_body=UserRegisterSerializer,
        operation_description="Create a new User",
        tags=['Account'],
        responses={201: UserRegisterSerializer(many=False)}
    )
    # Представление для регистрации пользователя
    def post(self, request, *args, **kwargs):
        serializer = UserRegisterSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserLoginView(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        request_body=UserLoginSerializer,
        operation_description="Login User",
        tags=['Account'],
        responses={201: UserLoginSerializer(many=False)}
    )
    def post(self, request, *args, **kwargs):
        print(request.data)
        serializer = UserLoginSerializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        refresh = RefreshToken.for_user(user)

        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }, status=status.HTTP_200_OK)


class UserUpdateView(APIView):
    # Представление для обновления профиля пользователя
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get user's detail",
        tags=['Account'],
        responses={200: UserDetailSerializer(many=True)}
    )
    def get(self, request):
        user = request.user
        serializer = UserDetailSerializer(user)
        return Response(serializer.data)

    @swagger_auto_schema(
        request_body=UserRegisterSerializer,
        operation_description="Update a User's details",
        tags=['Account'],
        responses={201: UserRegisterSerializer(many=False)}
    )
    def put(self, request):
        user = request.user
        serializer = UserRegisterSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserLogoutView(APIView):
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        operation_description="Logout User",
        tags=['Account'],
        responses={205: "Вы успешно вышли из системы."}
    )
    def post(self, request):
        try:
            refresh_token = request.data['refresh']
            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response({"detail": "Вы успешно вышли из системы."}, status=status.HTTP_205_RESET_CONTENT)
        except Exception as e:
            return Response({"detail": "Невозможно выйти из системы."}, status=status.HTTP_400_BAD_REQUEST)