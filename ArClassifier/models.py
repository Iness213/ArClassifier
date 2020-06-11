from django.db import models


class User(models.Model):
    login = models.CharField(max_length=12)
    password = models.CharField()


class Project(models.Model):
    project_name = models.CharField(max_length=200, unique=True)
    project_date = models.DateTimeField('creation date')
    project_description = models.CharField(max_length=300)
    user = models.ForeignKey(User, on_delete=models.CASCADE)


class Dataset(models.Model):
    dataset_path = models.CharField(max_length=40)

#class Result(models.Model):



