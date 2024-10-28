from rest_framework import serializers
from django.contrib.auth import get_user_model, authenticate

from apps.img_blocks.models import ImageModel

# Получаем модель пользователя
User = get_user_model()


class UserRegisterSerializer(serializers.ModelSerializer):
    uuid = serializers.UUIDField(required=False, write_only=True)
    password = serializers.CharField(write_only=True)

    class Meta:
        model = get_user_model()
        fields = ['username', 'password', 'email', 'phone', 'first_name', 'last_name', 'uuid']

    def create(self, validated_data):
        # Remove password from validated data to create user without setting the password initially
        password = validated_data.pop('password')

        # Create the user
        user = get_user_model().objects.create(**validated_data)

        # Set the password and save the user to hash the password
        user.set_password(password)
        user.save()

        uuid = validated_data.get('uuid', None)
        if uuid:
            try:
                # Fetch the ImageModel instance and update the user_identifier
                image_model = ImageModel.objects.get(uuid=uuid)
                image_model.user_identifier = user
                image_model.save()
            except ImageModel.DoesNotExist:
                # Handle the case where the ImageModel instance does not exist
                raise serializers.ValidationError("Image with the provided UUID does not exist.")

        return user

    def update(self, instance, validated_data):
        # Обновление данных пользователя
        instance.username = validated_data.get('username', instance.username)
        instance.email = validated_data.get('email', instance.email)
        instance.phone = validated_data.get('phone', instance.phone)
        instance.first_name = validated_data.get('first_name', instance.first_name)
        instance.last_name = validated_data.get('last_name', instance.last_name)
        instance.save()
        return instance


class UserLoginSerializer(serializers.Serializer):
    phone = serializers.CharField(max_length=15)
    password = serializers.CharField(
        label="Пароль",
        style={'input_type': 'password'},
        trim_whitespace=False
    )
    uuid = serializers.UUIDField(required=False, write_only=True)

    def validate(self, attrs):
        phone = attrs.get('phone')
        password = attrs.get('password')

        if phone and password:
            user = authenticate(request=self.context.get('request'),
                                phone=phone, password=password)

            if not user:
                msg = 'Невозможно войти с предоставленными учетными данными.'
                raise serializers.ValidationError(msg, code='authorization')
        else:
            msg = 'Необходимо включить "phone" и "password".'
            raise serializers.ValidationError(msg, code='authorization')

        attrs['user'] = user
        uuid = attrs.get('uuid', None)
        print(uuid,8)
        if uuid:
            try:
                # Fetch the ImageModel instance and update the user_identifier
                image_model = ImageModel.objects.get(uuid=uuid)
                print(image_model.user_identifier,2)
                image_model.user_identifier = user.uuid
                image_model.save()
                print(image_model.user_identifier,1)
            except ImageModel.DoesNotExist:
                # Handle the case where the ImageModel instance does not exist
                raise serializers.ValidationError("Image with the provided UUID does not exist.")
        return attrs


class UserDetailSerializer(serializers.ModelSerializer):

    class Meta:
        model = User
        fields = ['uuid', 'username', 'first_name', 'last_name', 'phone']
