from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from apps.account.models import CustomUser

class CustomUserAdmin(UserAdmin):
    # Определение полей для отображения и редактирования в админке
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Личная информация', {'fields': ('first_name', 'last_name', 'email', 'phone')}),
        ('Разрешения', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Важные даты', {'fields': ('last_login', 'date_joined')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'phone', 'password1', 'password2'),
        }),
    )
    # Настройка отображаемых колонок в списке пользователей
    list_display = ('uuid', 'username', 'email', 'first_name', 'last_name', 'phone', 'is_staff')
    search_fields = ('username', 'first_name', 'last_name', 'email', 'phone')
    ordering = ('username',)

# Регистрация модели CustomUser в админке
admin.site.register(CustomUser, CustomUserAdmin)
