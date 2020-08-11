from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.index, name='home'),
    path('datasets/', views.datasets, name='datasets'),
    path('projects/', views.projects, name='projects'),
    path('classification/', views.classification, name='classification'),
    path('savedResults/', views.saved_results, name='savedResults'),
    path('help/', views.help, name='help'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('forgotpass', views.forgotpass_view, name='forgotpass'),
]
