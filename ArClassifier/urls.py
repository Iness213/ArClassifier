from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('/login', views.login, name='login'),
    path('/signup', views.signup, name='signup'),
    path('/logout', views.logout, name='logout'),
    path('/forgot-password', views.forgotPassword, name='forgotPassword'),
    path('/reset-password', views.resetPassword, name='resetPassword'),
    path('/profile', views.profile, name='profile'),
    path('/library', views.library, name='library'),
    path('/myProjects', views.library, name='myProjects'),
    path('/project/<int:idProject>', views., name='project'),
]