from django.contrib.auth.models import AbstractUser
from django.db import models


class MyUser(AbstractUser):
    is_banned = models.BooleanField(default=False)

    def ban(self):
        self.is_banned = True
        self.save()


class Project(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()
    owner = models.ForeignKey(MyUser, related_name='user_owner', on_delete=models.CASCADE)


class Dataset(models.Model):
    name = models.CharField(max_length=255)
    path = models.CharField(max_length=255)
    project = models.ForeignKey(Project, related_name='project_project', on_delete=models.CASCADE)


class Keyword(models.Model):
    word = models.CharField(max_length=255)
    dataset = models.ForeignKey(Dataset, related_name='dataset_dataset', on_delete=models.CASCADE)
