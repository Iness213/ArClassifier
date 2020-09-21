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
        if os.path.isfile(self.path):
            os.remove(self.path)
        super(Dataset, self).delete(*args, **kwargs)


class TrainingSet(models.Model):
    name = models.CharField(max_length=255)
    path = models.CharField(max_length=255)


class Classification(models.Model):
    training_set = models.ForeignKey(TrainingSet, related_name='classification_trainingset', on_delete=models.CASCADE)
    file = models.ForeignKey(Dataset, related_name='file_classification', on_delete=models.CASCADE)
    project = models.ForeignKey(Project, related_name='project_classification', on_delete=models.CASCADE)
    algorithm = models.CharField(max_length=255, default="SVM + KNN + Naive Bayes")
    k_value = models.CharField(max_length=255, default=None, blank=True, null=True)
    type = models.CharField(max_length=255, default="Classification")
    creation_date = models.DateField()

    def save(self, *args, **kwargs):
        self.creation_date = datetime.now()
        super(Classification, self).save(*args, **kwargs)


class Result(models.Model):
    category = models.CharField(max_length=255)
    classification = models.ForeignKey(Classification, related_name='result_classification', on_delete=models.CASCADE)


class Keyword(models.Model):
    word = models.CharField(max_length=255)
    result = models.ForeignKey(Result, related_name='result_keyword', on_delete=models.CASCADE)


class Metric(models.Model):
    algorithm = models.CharField(max_length=255)
    accuracy = models.CharField(max_length=255)
    recall = models.CharField(max_length=255)
    precision = models.CharField(max_length=255)
    f1_score = models.CharField(max_length=255)
