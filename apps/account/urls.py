from django.urls import path
from apps.account.views import *
from apps.account.views import *


urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', UserLoginView.as_view(), name='login'),
    path('logout/', UserLogoutView.as_view(), name='logout'),
    path('profile/', UserUpdateView.as_view(), name='profile-update'),
]