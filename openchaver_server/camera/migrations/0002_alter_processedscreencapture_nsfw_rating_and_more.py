# Generated by Django 4.0.4 on 2022-05-17 04:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('camera', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='processedscreencapture',
            name='nsfw_rating',
            field=models.JSONField(blank=True, default=0, null=True),
        ),
        migrations.AlterField(
            model_name='processedscreencapture',
            name='skin_rating',
            field=models.JSONField(blank=True, default=0, null=True),
        ),
    ]
