# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render, HttpResponse
from django.http import HttpResponseRedirect
from models import NLPHandler
import json
import traceback
import copy
import logging
logger = logging.getLogger('save')
logger_error = logging.getLogger('wrong')

nlp = NLPHandler()

def get_web_data(request, label):
    sseq = ''
    if request.method == 'GET':
        sseq = request.GET[label]
    else:
        sseq = request.POST[label]
    return sseq


def test(request):
    return render(request, 'index.html')

def process_data(request):
    try:
        seq = get_web_data(request, 'seq')
        task = get_web_data(request, 'task')
        if len(seq.strip()) == 0:
            res = 'NULL'
        else:
            if task == 'scene':
                res = nlp(seq,'scene')
                res = json.dumps(res, ensure_ascii=False)
            else:
                res = nlp(seq, task)
                res = json.dumps(res, ensure_ascii=False)
                res = res.encode('utf-8')
                res = HttpResponse(res)
                res["Access-Control-Allow-Origin"] = "*"
                res["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
                res["Access-Control-Max-Age"] = "1000"
                res["Access-Control-Allow-Headers"] = "*"
        return res
    except BaseException as e:
        logger_error.info(seq)
        logger_error.info('code error info:\n%s' %
                          traceback.format_exc())
        return HttpResponse('NLP not response,please Check error log')

def url_way(request):
    try:
        seq = get_web_data(request, 'sentence')
        task = get_web_data(request, 'task')
        if len(seq.strip()) >= 1:
            res = nlp(seq, task)
            res = json.dumps(res, ensure_ascii=False)
            logger.info(res)
            res = res.encode('utf-8')
            return HttpResponse(res)
        else:
            logger.info("Sorry,Check your input!!")
            return HttpResponse("Sorry,Check your input!!")
    except BaseException as e:
        logger_error.info(seq)
        logger_error.info('code error info:\n%s' %
                          traceback.format_exc())
        return HttpResponse('NLP not response,please Check error log')


def url_way_test(request):
    try:
        seq = get_web_data(request, 'sentence')
        task = get_web_data(request, 'task')
        if len(seq.strip()) >= 1:
            if task == 'scene':
                res = nlp(seq,'scene')
            else:
                res = nlp(seq, task)
            res = json.dumps(res, ensure_ascii=False)
            logger.info(res)
            res = res.encode('utf-8')
            return HttpResponse(res)
        else:
            logger.info("Sorry,Check your input!!")
            return HttpResponse("Sorry,Check your input!!")
    except BaseException as e:
        logger_error.info(seq)
        logger_error.info('code error info:\n%s' %
                          traceback.format_exc())
        return HttpResponse('NLP not response,please Check error log')


def url_way_scene(request):
    try:
        seq = get_web_data(request, 'sentence')
        if len(seq.strip()) >= 1:
            res = nlp(seq,'scene')
            res = json.dumps(res, ensure_ascii=False)
            logger.info(res)
            res = res.encode('utf-8')
            return HttpResponse(res)
        else:
            logger.info("Sorry,Check your input!!")
            return HttpResponse("Sorry,Check your input!!")
    except BaseException as e:
        logger_error.info(seq)
        logger_error.info('code error info:\n%s' %
                          traceback.format_exc())
        return HttpResponse('NLP not response,please Check error log')


def url_chat(request):
    try:
        seq = get_web_data(request, 'sentence')
        if len(seq.strip()) == 0:
            res = 'NULL'
        else:
            res = nlp(seq, 'chat')
            logger.info(res)
            res = res.encode('utf-8')
        return HttpResponse(res)
    except BaseException as e:
        logger_error.info(seq)
        logger_error.info('code error info:\n%s' %
                          traceback.format_exc())
        return HttpResponse('NLP not response,please Check error log')
