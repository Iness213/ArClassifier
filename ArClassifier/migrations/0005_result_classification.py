# Generated by Django 3.1 on 2020-08-21 18:22

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('ArClassifier', '0004_auto_20200818_2154'),
    ]

    operations = [
        migrations.AddField(
            model_name='result',
            name='classification',
            field=models.ForeignKey(default='0', on_delete=django.db.models.deletion.CASCADE, related_name='result_classification', to='ArClassifier.classification'),
            preserve_default=False,
        ),
    ]
