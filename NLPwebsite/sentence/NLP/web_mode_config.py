# model exclude conditions
TV_MODE = 'tv'  # tv mode
PHONE_MODE = 'phone'  # phone mode
NGTV_MODE = 'ngtv'
# for open-domain chatting
CHAT_MODE = 'chat'

CONFIG_MODE = TV_MODE  # default mode is tv
# EC model switsh
USED_EC = True
# config path
import os
CONFIG_PATH = os.getcwd() + '/sentence/NLP/conf/%s.command.conf' % (CONFIG_MODE)
# NLP version number, default version number is tv
version = 'v1.6.3'
if CONFIG_MODE == NGTV_MODE:
    version = 'v1.1.0'
if CONFIG_MODE == PHONE_MODE:
    version = 'v1.1.21'
if CONFIG_MODE == CHAT_MODE:
    version = 'v0.0.1'
