# -*- coding: utf-8 -*-
"""Generate ground truth videos"""

import cv2
import glob

msasl_glosses = ['all/','alone/','apple/','bathtub/','depressed/','culture/','crazy/','autumn/','afternoon/','beautiful/',
            'boy/','busy/','cake/','camp/','cat/','cold/','dinner/','discuss/','divorce_8543/','do_6301/',
            'draw_2756/', 'dress/', 'drive_5036/','earn/','ears/','eight_703/','europe/',
            'fix/','front/','future_2547/','girl/','group_0/','hate/','hearing_5505/','hill/','hotel_0/','hurry/',
            'hurt_1576/','japan/','judge/','junior/','jump/','late/','make/','pull/','rainbow_174/','slow_0/','rock/','river/','several/']
gl_glosses = ['a/','abend/','aber/','amt/','biene/','bier/','bikini/','ecke/','ehe/','gruppe/',
            'gummi/','gurke/','komma/','kondom/','liste/','loch/','locker/','mundbild/','musical/','musik/',
            'nie/','niere/','nn/','regional/','reh/','reihe/','seife/','seite/','tor/','traege/',
            'wirklich/','wissen/','wild/','zeh/','zeuge/','zeitung/','toben/','titel/','somalia/','software/',
            'sonne/','pfeffer/','petzen/','monat/','mitte/','lernen/','klavier/','klauen/','kampf/','hoden/']
benchmark = ['college_6283/', 'happy_2595/', '50/', 'alarm_166/', 'cat_1439/', 'africa_381/', 'all_1869/', 'break_2348/', 'black_1743/',
            'clear_1734/', 'cool_6271/', 'asl/', 'enjoy_868/', 'grapes_10367/', 'country_6008/', 'australia_1645/', 'government_6776/', 'both_2575/', 
            'africa_381/', 'brother_2446/', 'chocolate_8509/', 'half_7041/', 'couch_3742/', 'clothes_126/', 'garage_14435/', 'excited_2461/', 'east/', 
            'cent_4815/', 'brother_6512/', 'apple_10094/', 'fire_4504/', 'believe_318/', 'get up_14113/', 'funny_5818/', 'eat/', 'athlete_4872/', 'football_1161/', 
            'flat tire_0/', 'every week/', 'fish_701/', 'hearing help_7086/', 'carrot_6285/', 'deer_9226/', 'believe_14808/', 'day_341/', 'exercise_5092/', 'around_765/', 
            'every morning_6547/', 'hearing_5328/', '21_3568/', 'father_9912/', 'cochlear implant_6410/', 'chicken_665/', 'good_9837/', 'every morning_6547/', 'beer_4696/', 
            'college_6113/', 'center school_18034/', 'bowling_7479/', 'girl_12234/', 'dorm_10404/', 'cheese_3078/', 'beautiful_6520/', 'cousin_3266/', 'discuss_4256/', 
            'cherry/', 'family_2236/', 'army/', 'college_6283/', 'get up_14113/', 'bread_5723/', 'eat_1138/', 'alarm_933/', 'corn_9480/', 'bear_762/', 'college_0/', 
            'hearing_5328/', 'chocolate_8429/', 'for_12063/', 'bus_201/', 'cat_5367/', 'different_3787/', 'credit card_8350/', 'funny_5069/', 'fish_1554/', 
            'finland_10225/', 'doctor_2869/', 'ears_7867/', 'frustrated_11380/', 'doctor_8225/', 'chicken_8615/', 'deaf_1569/', 'door_2803/', 'baseball_5471/', 
            'gymnastics_5961/', 'angry_2879/', 'clear_1734/', 'give_9577/', 'butter_26892/', 'good afternoon_5143/', 'born_2917/', 'friend_3959/', 'day_647/', 'chicken_721/', 
            'beach_2366/', 'beer_4047/', 'cheese_7734/', 'cow_3246/', 'autumn in love_3086/', 'actor_432/', 'bird_2899/', 'grey_897/', 'beard_4243/', 'alarm_10938/', 'aunt_1655/', 
            'grandmother_1122/', 'fail_4604/', 'blanket_9940/', 'hearing help_1853/', 'earn_36959/', 'free_13875/', 'country_18564/', 'corn_2062/', 'grandmother_14372/', 'cute_7317/', 
            'acquire/', 'drink/', 'cow_2585/', 'cracker_4759/', 'bacon_3193/', 'arrogant/', 'friend_3144/', 'computer_3997/', 'hamburger_3336/', 'brother_3351/', 'black/', 'class_6041/', 
            'go_9750/', 'email_0/', 'boy_3588/', 'cherry_8026/', '21_3568/', 'free_920/', 'full_2526/', 'apple_10094/', 'door_5067/', 'drive_375/', 'friendly_20862/', 'friend_3144/', 
            "can't_2323/", 'butterfly_1271/', 'but_7227/', 'bird_3089/', 'deer_9226/', 'gymnastics_5812/', 'bowl_577/', 'eighteen_0/', 'girlfriend_4776/', 'everyday_6907/', 
            'hearing help_7086/', '22_3632/', 'give_2404/', 'cousin_120/', 'halloween_10233/', 'fat_4673/', 'hamburger_12881/', 'careful_580/', 'asl/', 'early_9001/', 
            'blue_1777/', 'chicken_3865/', 'cereal_7561/', 'chemistry_1246/', 'beautiful_22376/', 'flirt_10458/', 'eleven_2855/', 'alarm/', 'gray_1530/', 'golf_3413/', 
            'half hour_12309/', 'giraffe_12532/', 'flag_10235/', 'doctor_7263/', 'grandfather_13952/', 'annoy_656/', 'english_2195/', 'every morning_11309/', 
            'beautiful_194/', 'draw_645/', 'europe_9924/', 'bowling_7479/', 'elementary school_2769/', 'easy_2048/', 'fifteen_823/', 'fish_791/', 'bad_3605/', 
            'cost_7987/', 'frog_2661/', '29_3979/', 'computer_5526/']

for g in benchmark:
    video_frames = []
    for filename in sorted(glob.glob("./data/" + g + "*.jpg")):
        img = cv2.imread(filename)
        img = cv2.resize(img, (256,256), interpolation = cv2.INTER_CUBIC)
        video_frames.append(img)

    out = cv2.VideoWriter("./videos/ground_truth/" + g[:-1] + ".avi", cv2.VideoWriter_fourcc(*'DIVX'), 25, (256,256))

    for i in range(len(video_frames)):
        out.write(video_frames[i])
    out.release()