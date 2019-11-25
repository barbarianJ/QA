"""
Django settings for NLPwebsite project.

Generated by 'django-admin startproject' using Django 1.11.4.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.11/ref/settings/
"""
import os
import logging
import django.utils.log
import logging.handlers
# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#make log dir
LOG_PATH = os.path.join(BASE_DIR, 'log')
if not os.path.isdir(LOG_PATH):
    os.mkdir(LOG_PATH)
LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
       'standard': {
            'format': '%(asctime)s [%(name)s:%(lineno)d] [%(module)s:%(funcName)s] : %(message)s'},
        'debug_format': {
            'format': '%(message)s'}
    },
    'filters': {
    },
    'handlers': {
        'mail_admins': {
            'level': 'ERROR',
            'class': 'django.utils.log.AdminEmailHandler',
            'include_html': True,
        },
        'normal_log': {
            'level':'INFO',
            'class':'logging.handlers.RotatingFileHandler',
            'filename':'%s/normal_log.txt' % LOG_PATH,     #log out file
            'maxBytes': 1024*1024*5,                #log file size 
            'backupCount': 5,                       #backup 
            'formatter':'standard',                 #log format
        },
        'error_log': {
            'level':'INFO',
            'class':'logging.handlers.RotatingFileHandler',
            'filename':'%s/error_log.txt' % LOG_PATH,     #log out file
            'maxBytes': 1024*1024*5,                #log file size 
            'backupCount': 5,                       #backup 
            'formatter':'standard',                 #log format
        },
        'test_save': {
            'level':'INFO',
            'class':'logging.handlers.RotatingFileHandler',
            'filename':'%s/save_log.txt' % LOG_PATH,     #log out file
            'maxBytes': 1024*1024*5,                #log file size 
            'backupCount': 5,                       #backup 
            'formatter':'debug_format',                 #log format
        },
        'console':{
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'debug_format'
        },
    },
    'loggers': {
        'wrong': {
            'handlers': ['error_log'],
            'level': 'INFO'
        },
        'save': {
            'handlers': ['normal_log'],
            'level': 'INFO'
        },
        'debug_save': {
            'handlers': ['test_save'],
            'level': 'INFO'
        },
        'debug': {
            'handlers': ['console'],
            'level': 'INFO'
        }
    },
}

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.11/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'tj2+$vi3-4&l1tl#=i8%5!_^t&b1=7q8!ii+=^3rh92fyv#q7a'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*',]


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'sentence',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    #'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'NLPwebsite.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR,'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'NLPwebsite.wsgi.application'


# Database
# https://docs.djangoproject.com/en/1.11/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': None,
    }
}
# Password validation
# https://docs.djangoproject.com/en/1.11/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/1.11/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.11/howto/static-files/

STATIC_URL = '/static/'
HERE = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.join(HERE,'../')
STATICFILES_DIRS = (
    os.path.join(HERE,'static/'),
    )