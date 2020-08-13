from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.index, name='home'),

    path('projects/', views.projects, name='projects'),
    path('project/<int:id>', views.project, name='project'),
    path('addProject/', views.add_project, name='addProject'),
    path('deleteProject/<int:id>', views.delete_project, name='deleteProject'),

    path('datasets/', views.datasets, name='datasets'),
    path('deleteFile/<int:id>', views.delete_file, name='deleteFile'),

    path('feed_text/', views.feed_text, name='feed_text'),
    path('upload_file/<int:id>', views.upload_file, name='upload_file'),
    path('classification/', views.classification, name='classification'),
    path('savedResults/', views.saved_results, name='savedResults'),
    path('help/', views.help, name='help'),

    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('forgotpass', views.forgotpass_view, name='forgotpass'),
]
