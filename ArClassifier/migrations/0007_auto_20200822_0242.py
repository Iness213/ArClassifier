# Generated by Django 3.1 on 2020-08-22 01:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ArClassifier', '0006_auto_20200822_0053'),
    ]

    operations = [
        migrations.AlterField(
            model_name='classification',
            name='k_value',
            field=models.CharField(blank=True, default=None, max_length=255, null=True),
        ),
    ]
