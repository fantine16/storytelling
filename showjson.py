#-*-coding:utf-8-*-
import json

image_set_file='dataset/image_set.json'
story_set_file='dataset/story_set.json'
storytelling_file = 'dataset/storytelling.json'

image_set = json.load(open(image_set_file,'r'))
story_set = json.load(open(story_set_file,'r'))
storytelling = json.load(open(storytelling_file,'r'))

#key_l = list(story_set)
#value_l = story_set.values()

print('show json')