# Generated by Django 3.0 on 2020-04-27 19:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0012_auto_20200425_1722'),
    ]

    operations = [
        migrations.CreateModel(
            name='Inference',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(null=True, upload_to='')),
                ('prediction', models.CharField(max_length=200)),
            ],
        ),
    ]
