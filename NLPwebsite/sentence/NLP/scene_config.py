#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-25 13:15:51
# @Author  : Lee (lijingyang@tcl.com)

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


# the original source dir of datas
Original_corpus_dir = '/home/lee/workspace/cls-nlp_git/cls-nlp/cls_scene/corpus_1120/'

# the preprocess data with ner_keyword source dir of datas
#Preprocess_corpus_dir = '/home/lee/workspace/cls-nlp_git/cls-nlp/cls_scene/corpus_new/'

# the corpus dir for model
Corpus_dir = './corpus/'

"""
Define labels and datas for Demo v1
   for item in tup:
     item[0] included filesname
     item[1] represent numic label
     item[2] represent label name
"""

# this tup is defined for Original_corpus_dir

Data_tup_scene = [(['play_control.data', 'tv_ui.data',
                    'tv_setting_and_state_query.data',
                    'history.data', 'control_home.data', 'control_tv.data',
                    'household_appliance_control.data', 'scene_find.data'], 0,
                   'control'),

                  (['app_basic_control.data', 'app_search.data',
                    'app_download_install_uninstall.data',
                    'mixed_app_control.data'], 1, 'app_control'),

                  (['film.data', 'tv_play.data', 'programme.data', 'normal_video.data',
                    'traditional_opera.data', 'acrobatics.data',
                    'short_sketch.data', 'comic_dialogue.data', 'education.data',
                    'cartoon.data', 'sports.data', 'mixed_video.data'], 2, 'video'),

                  (['basic_weather.data', 'basic_rate.data',
                    'disaster_warn.data', 'mixed_weather.data'], 3, 'weather'),

                  (['channel_switch.data'], 4, 'channel_switch'),

                  (['image_scene_interactive.data', ], 5, 'image_interactive'),

                  (['math_operation.data'], 6, 'math_operation'),

                  (['joke.data'], 7, 'disport'),

                  (['time_query.data', 'time_zone_query.data'], 8, 'time_query'),

                  (['humanity_encyclopedia.data', 'science_encyclopedia.data',
                    'media_encyclopedia.data', 'music_encyclopedia.data',
                    'mixed_encyclopedia_information.data',
                    'baidu_encyclopedia.data', 'zhihu_encyclopedia.data',
                    'add_media_encyclopedia.data'], 9, 'baike'),

                  (['normal_news.data', 'industry_news.data',
                    'joy_news.data', 'mixed_info_query_news.data'], 10, 'info_news'),

                  (['stock_fund_information.data'], 11, 'info_stock&fund'),

                  (['translation.data'], 12, 'translate'),

                  (['currency_converter.data', 'unit_conversion.data',
                    'mixed_converter.data'], 13, 'converter'),

                  (['karaoke.data'], 14, 'karaoke'),

                  (['word.data'], 15, 'words'),

                  (['music_audio.data', 'fm_audio.data'], 16, 'audio'),

                  (['map_server.data'], 17, 'map_server'),

                  (['ticket_server.data'], 18, 'ticket_server'),

                  (['order_server.data'], 19, 'order_server'),

                  (['third_info_query.data'], 20, 'third_query'),

                  (['multi-turn_dialogue.data',
                    'tv_ui_dialogue.data',
                    'play_control_dialogue.data',
                    'tv_setting_and-state_query_dialogue.data',
                    'video_multi_dialogue.data']
                   , 21, 'multi_dialogue'),

                  (['unknown_pos.data', 'unknown_new.data', 'chat.data', 'mixed_unknown.data'], 22, 'unknown'),

                  (['universal_search.data'], 23, 'search'),

                  (['couplet.data'], 24, 'couplet'),

                  (['poetry.data'], 25, 'poetry'),

                  (['re_dialogue.data', 'add_relation.data', 'remember_relation.data'],
                   26, 'relation_dialogue'),

                  (['album.data', 'image_qr.data'], 27, 'album'),

                  (['request_operation.data', 'request_repeat.data'], 28, 'help'),

                  ]

# This tup is define for corpus_Generalization/map_server
Data_tup_map = [(['restaurant_recommend.data'], 0, 'map_restaurant'),

               (['hotel_recommend.data'], 1, 'map_hotel'),

               (['scenic_spots_recommend.data'], 2, 'map_scenic'),

               (['cinema_recommend.data'], 3, 'map_cinema'),

               (['map.data','navigation.data','traffic.data'], 4, 'map_basic')
               ]

# This tup is define for corpus_Generalization/ticket
Data_tup_ticket = [(['air_tickets.data','train_tickets.data'], 0, 'ticket_travel'),

               (['book_scenic_spots_ticket.data'], 1, 'ticket_scenic'),

               (['book_movie_tickets.data'], 2, 'ticket_cinema')
               ]

# This tup is define for corpus_Generalization/order
Data_tup_order = [(['takeout_food.data','takeout_fresh.data'], 0, 'order_food'),

               (['takeout_shopping.data']	, 1, 'order_goods'),

               (['takeout_restaurant.data'], 2, 'order_restaurant'),

               (['takeout_hotel.data'], 3, 'order_hotel')
               ]


# This tup is define for corpus_Generalization/audio
Data_tup_audio = [(['music_audio.data'], 0, 'music'),

                  (['fm_audio.data'], 1, 'fm_audio')
                  ]
# This tup is define for corpus_Generalization/baike
Data_tup_baike = [(['media_encyclopedia.data'], 0, 'media_baike'),

               (['humanity_encyclopedia.data', 'science_encyclopedia.data', 'mixed_encyclopedia_information.data',
                 'baidu_encyclopedia.data', 'zhihu_encyclopedia.data','music_encyclopedia.data'], 1, 'basic_baike')
               ]

# Data_tup_binary = [(['negative_sample.data'], 0, 'negative'),
#
#                   (['positive_sample.data'], 1, 'positive'),]

# This tup is define for ngtv
Data_tup_ngtv = [(['play_control.data', 'tv_ui.data',
                    'tv_setting_and_state_query.data',
                    'history.data', 'control_home.data', 'control_tv.data',
                    'household_appliance_control.data', 'scene_find.data'], 0,
                   'control'),

                  (['app_basic_control.data', 'app_search.data',
                    'app_download_install_uninstall.data',
                    'mixed_app_control.data'], 1, 'app_control'),

                  (['film.data', 'tv_play.data', 'programme.data', 'normal_video.data',
                    'traditional_opera.data', 'acrobatics.data',
                    'short_sketch.data', 'comic_dialogue.data', 'education.data',
                    'cartoon.data', 'sports.data', 'mixed_video.data'], 2, 'video'),

                  (['basic_weather.data', 'basic_rate.data',
                    'disaster_warn.data', 'mixed_weather.data'], 3, 'weather'),

                  (['channel_switch.data'], 4, 'channel_switch'),

                  (['image_scene_interactive.data', ], 5, 'image_interactive'),

                  (['math_operation.data'], 6, 'math_operation'),

                  (['joke.data'], 7, 'disport'),

                  (['time_query.data', 'time_zone_query.data'], 8, 'time_query'),

                  (['humanity_encyclopedia.data', 'science_encyclopedia.data',
                    'media_encyclopedia.data', 'music_encyclopedia.data',
                    'mixed_encyclopedia_information.data',
                    'baidu_encyclopedia.data', 'zhihu_encyclopedia.data',
                    'add_media_encyclopedia.data'], 9, 'baike'),

                  (['normal_news.data', 'industry_news.data',
                    'joy_news.data', 'mixed_info_query_news.data'], 10, 'info_news'),

                  (['stock_fund_information.data'], 11, 'info_stock&fund'),

                  (['translation.data'], 12, 'translate'),

                  (['currency_converter.data', 'unit_conversion.data',
                    'mixed_converter.data'], 13, 'converter'),

                  (['karaoke.data'], 14, 'karaoke'),

                  (['word.data'], 15, 'words'),

                  (['music_audio.data', 'fm_audio.data'], 16, 'audio'),

                  (['map_server.data'], 17, 'map_server'),

                  (['ticket_server.data'], 18, 'ticket_server'),

                  (['order_server.data'], 19, 'order_server'),

                  (['third_info_query.data'], 20, 'third_query'),

                  (['multi-turn_dialogue.data',
                    'tv_ui_dialogue.data',
                    'play_control_dialogue.data',
                    'tv_setting_and-state_query_dialogue.data',
                    'video_multi_dialogue.data']
                   , 21, 'multi_dialogue'),

                  (['unknown_pos.data', 'unknown_new.data', 'chat.data', 'mixed_unknown.data'], 22, 'unknown'),

                  (['universal_search.data'], 23, 'search'),

                  (['couplet.data'], 24, 'couplet'),

                  (['poetry.data'], 25, 'poetry'),

                  (['re_dialogue.data', 'add_relation.data', 'remember_relation.data'],
                   26, 'relation_dialogue'),

                  (['album.data','image_qr.data'], 27, 'album'),

                  (['request_operation.data', 'request_repeat.data'], 28, 'help'),

                  ]
