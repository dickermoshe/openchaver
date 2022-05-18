# Generated by Django 4.0.4 on 2022-05-18 00:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('camera', '0009_alter_screencapture_image'),
    ]

    operations = [
        migrations.RenameField(
            model_name='screencapture',
            old_name='average_safe',
            new_name='average_nsfw',
        ),
        migrations.AddField(
            model_name='screencapture',
            name='max_nsfw',
            field=models.FloatField(blank=True, null=True),
        ),
    ]