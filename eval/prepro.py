import json

storyset_file = '../dataset/story_set.json'

val_file = './annotations/anno_val.json'
test_file = './annotations/anno_test.json'
train_file = './annotations/anno_train.json'

val = {}
val['annotations'] = []
test = {}
test['annotations'] = []
train = {}
train['annotations'] = []


story_set = json.load(open(storyset_file, 'r'))

for key in story_set:
    term = {}
    term['image_id'] = key.encode('utf-8')
    term['caption'] = ''
    for text in story_set[key]['text']:
        term['caption'] = term['caption'] + text.encode('utf-8')
    if story_set[key]['split'] == 'train':
        train['annotations'].append(term)
    if story_set[key]['split'] == 'test':
        test['annotations'].append(term)
    if story_set[key]['split'] == 'val':
        val['annotations'].append(term)

json.dump(val, open(val_file, 'w'))
json.dump(test,open(test_file, 'w'))
json.dump(train,open(train_file, 'w'))
