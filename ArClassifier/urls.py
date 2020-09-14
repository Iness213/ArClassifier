from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.index, name='home'),

    path('projects/', views.projects, name='projects'),
    path('project/<int:id>', views.project, name='project'),
    path('addProject/', views.add_project, name='addProject'),
    path('deleteProject/<int:id>', views.delete_project, name='deleteProject'),

    path('file/<int:id>', views.file, name='file'),
    path('deleteFile/<int:id>', views.delete_file, name='deleteFile'),
    path('dowload/<int:id>', views.download_file, name='download_file'),
    path('feed_text/<int:id>', views.feed_text, name='feed_text'),
    path('upload_file/<int:id>', views.upload_file, name='upload_file'),
    path('preprocess_text/<int:id>', views.preprocess_text, name='preprocess_text'),


    path('classification/<int:id>', views.classification, name='classification'),
    path('sentiment/<int:id>', views.sentiment, name='sentiment'),

    path('result/<int:id>', views.result, name='result'),

    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('forgotpass', views.forgotpass_view, name='forgotpass'),
]
