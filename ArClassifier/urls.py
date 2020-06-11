from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('/login', views.login, name='login'),
    path('/signup', views., name=''),
    path('/logout', views., name=''),
    path('/forgot-password', views., name=''),
    path('/reset-password', views., name=''),
    path('/profile', views., name=''),
    path('/projects', views., name=''),
    path('/project/<int:idProject>', views., name=''),
    path('/project/<int:idProject>/datasets', views., name=''),
]