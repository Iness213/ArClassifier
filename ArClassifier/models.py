import os

from django.contrib.auth.models import AbstractUser
from django.db import models
from datetime import datetime


class MyUser(AbstractUser):
    is_banned = models.BooleanField(default=False)

    def ban(self):
        self.is_banned = True
        self.save()


class Project(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()
    creation_date = models.DateField()
    owner = models.ForeignKey(MyUser, related_name='user_owner', on_delete=models.CASCADE)

    def save(self, *args, **kwargs):
        self.creation_date = datetime.now()
        super(Project, self).save(*args, **kwargs)


class Dataset(models.Model):
    name = models.CharField(max_length=255)
    path = models.CharField(max_length=255)
    creation_date = models.DateField()
    project = models.ForeignKey(Project, related_name='project_project', on_delete=models.CASCADE)

    def save(self, *args, **kwargs):
        self.creation_date = datetime.now()
        super(Dataset, self).save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        os.remove(self.path)
        super(Dataset, self).delete(*args, **kwargs)


class Keyword(models.Model):
    word = models.CharField(max_length=255)
    dataset = models.ForeignKey(Dataset, related_name='dataset_dataset', on_delete=models.CASCADE)


class Result(models.Model):
    category = models.CharField(max_length=255)
    keyword = models.ForeignKey(Keyword, related_name='keyword_keyword', on_delete=models.CASCADE)
