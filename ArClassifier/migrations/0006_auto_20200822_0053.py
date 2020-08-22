# Generated by Django 3.1 on 2020-08-21 23:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ArClassifier', '0005_result_classification'),
    ]

    operations = [
        migrations.CreateModel(
            name='Metric',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('algorithm', models.CharField(max_length=255)),
                ('accuracy', models.CharField(max_length=255)),
                ('recall', models.CharField(max_length=255)),
                ('precision', models.CharField(max_length=255)),
                ('f1_score', models.CharField(max_length=255)),
            ],
        ),
        migrations.RemoveField(
            model_name='result',
            name='accuracy',
        ),
        migrations.RemoveField(
            model_name='result',
            name='f1_score',
        ),
        migrations.RemoveField(
            model_name='result',
            name='precision',
        ),
        migrations.RemoveField(
            model_name='result',
            name='recall',
        ),
    ]
