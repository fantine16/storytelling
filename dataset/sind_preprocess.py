#-*-coding:utf-8-*-
import json
import argparse
import os
import sys


def get_story_set(sis_test, sis_train, sis_val, image_set):

    print('计算story的字典...')
    story_set = {}
    for i, term in enumerate(sis_test['annotations']):
        story_id = term[0]['story_id'].encode('utf-8')
        if not story_set.has_key(story_id):
            story_set[story_id] = {}
            story_set[story_id]['img_num'] = 1
            story_set[story_id]['text'] = []
            story_set[story_id]['text'].append(term[0]['text'].encode('utf-8'))
            story_set[story_id]['imagename'] = []
            story_set[story_id]['imagename'].append(image_set[term[0]['photo_flickr_id'].encode('utf-8')]['imagename'])
            story_set[story_id]['split']='test'
            story_set[story_id]['image_id'] = []
            story_set[story_id]['image_id'].append(term[0]['photo_flickr_id'].encode('utf-8'))
        else:
            story_set[story_id]['img_num'] = 1+story_set[story_id]['img_num']
            story_set[story_id]['text'].append(term[0]['text'].encode('utf-8'))
            story_set[story_id]['imagename'].append(image_set[term[0]['photo_flickr_id'].encode('utf-8')]['imagename'])
            story_set[story_id]['image_id'].append(term[0]['photo_flickr_id'].encode('utf-8'))
    for i, term in enumerate(sis_train['annotations']):
        story_id = term[0]['story_id'].encode('utf-8')
        if not story_set.has_key(story_id):
            story_set[story_id] = {}
            story_set[story_id]['img_num'] = 1
            story_set[story_id]['text'] = []
            story_set[story_id]['text'].append(term[0]['text'].encode('utf-8'))
            story_set[story_id]['imagename'] = []
            story_set[story_id]['imagename'].append(image_set[term[0]['photo_flickr_id'].encode('utf-8')]['imagename'])
            story_set[story_id]['split'] = 'train'
            story_set[story_id]['image_id'] = []
            story_set[story_id]['image_id'].append(term[0]['photo_flickr_id'].encode('utf-8'))
        else:
            story_set[story_id]['img_num'] = 1 + story_set[story_id]['img_num']
            story_set[story_id]['text'].append(term[0]['text'].encode('utf-8'))
            story_set[story_id]['imagename'].append(image_set[term[0]['photo_flickr_id'].encode('utf-8')]['imagename'])
            story_set[story_id]['image_id'].append(term[0]['photo_flickr_id'].encode('utf-8'))
    for i, term in enumerate(sis_val['annotations']):
        story_id = term[0]['story_id'].encode('utf-8')
        if not story_set.has_key(story_id):
            story_set[story_id] = {}
            story_set[story_id]['img_num'] = 1
            story_set[story_id]['text'] = []
            story_set[story_id]['text'].append(term[0]['text'].encode('utf-8'))
            story_set[story_id]['imagename'] = []
            story_set[story_id]['imagename'].append(image_set[term[0]['photo_flickr_id'].encode('utf-8')]['imagename'])
            story_set[story_id]['split'] = 'val'
            story_set[story_id]['image_id'] = []
            story_set[story_id]['image_id'].append(term[0]['photo_flickr_id'].encode('utf-8'))
        else:
            story_set[story_id]['img_num'] = 1 + story_set[story_id]['img_num']
            story_set[story_id]['text'].append(term[0]['text'].encode('utf-8'))
            story_set[story_id]['imagename'].append(image_set[term[0]['photo_flickr_id'].encode('utf-8')]['imagename'])
            story_set[story_id]['image_id'].append(term[0]['photo_flickr_id'].encode('utf-8'))

    return story_set


def get_image_set(sis_test, sis_train, sis_val):

    print('计算SIND数据集全部图像的字典...')
    images_set = {}
    for term in sis_test['images']:
        image_id = term['id'].encode('utf-8')
        images_set[image_id] = {}
        try:
            url = term['url_o']
        except:
            url = term['url_m']
        url = url.encode('utf-8')
        images_set[image_id]['imagename'] = url.split('/')[-1]
        images_set[image_id]['split'] = 'test'
    for term in sis_train['images']:
        image_id=term['id'].encode('utf-8')
        images_set[image_id] = {}
        try:
            url = term['url_o']
        except:
            url = term['url_m']
        url = url.encode('utf-8')
        images_set[image_id]['imagename'] = url.split('/')[-1]
        images_set[image_id]['split'] = 'train'
    for term in sis_val['images']:
        image_id=term['id'].encode('utf-8')
        images_set[image_id] = {}
        try:
            url = term['url_o']
        except:
            url = term['url_m']
        url = url.encode('utf-8')
        images_set[image_id]['imagename'] = url.split('/')[-1]
        images_set[image_id]['split'] = 'val'
    return images_set


def refine_story_set(story_set):

    print('进一步处理story的字典，把包含无效图片标注的story项删去...')
    files_val = os.listdir('val')
    files_train = os.listdir('train')
    files_test = os.listdir('test')
    files = files_val+files_train+files_test
    story_set_refine = {}
    bar_length = 30
    for k,term in enumerate(story_set):
        flag = True
        for img in story_set[term]['imagename']:
            if img not in files:
                flag = False
        if flag:
            story_set_refine[term] = story_set[term]
        hashes = '#' * int(1.0 * k / len(story_set) * bar_length)
        spaces = ' ' * (bar_length - len(hashes))
        sys.stdout.write("\rPercent: [%s] %d%%" % (hashes + spaces, 100 * k / len(story_set)))
    return story_set_refine


def calclt_img(image_set,story_set):

    for img_id in image_set:
        image_set[img_id]['anno_num'] = 0
    for term in story_set:
        for img_id in story_set[term]['image_id']:
            image_set[img_id]['anno_num'] = image_set[img_id]['anno_num']+1

    img_stat = {}
    img_test_stat = {}
    img_train_stat = {}
    img_val_stat = {}
    anno_num = 0
    anno_test_num = 0
    anno_train_num = 0
    anno_val_num = 0
    img_num = 0
    img_test_num = 0
    img_train_num = 0
    img_val_num = 0
    img_test_valid_num = len(os.listdir('test'))
    img_train_valid_num = len(os.listdir('train'))
    img_val_valid_num = len(os.listdir('val'))
    img_valid_num = img_test_valid_num + img_train_valid_num + img_val_valid_num

    for term in image_set:
        count = image_set[term]['anno_num']
        anno_num = anno_num+count
        img_num = img_num+1
        if img_stat.has_key(count):
            img_stat[count] = img_stat[count]+1
        else:
            img_stat[count] = 1
        if image_set[term]['split'] == 'test':
            count = image_set[term]['anno_num']
            anno_test_num = anno_test_num+count
            img_test_num = img_test_num+1
            if img_test_stat.has_key(count):
                img_test_stat[count] = img_test_stat[count] + 1
            else:
                img_test_stat[count] = 1
        if image_set[term]['split'] == 'train':
            count = image_set[term]['anno_num']
            anno_train_num = anno_train_num+count
            img_train_num=img_train_num+1
            if img_train_stat.has_key(count):
                img_train_stat[count] = img_train_stat[count] + 1
            else:
                img_train_stat[count] = 1
        if image_set[term]['split'] == 'val':
            count = image_set[term]['anno_num']
            anno_val_num = anno_val_num+count
            img_val_num = img_val_num+1
            if img_val_stat.has_key(count):
                img_val_stat[count] = img_val_stat[count] + 1
            else:
                img_val_stat[count] = 1


    print('SIND数据集共有 %d 个图像，训练集大小是 %d，测试集大小是 %d，验证集大小是 %d' % (img_num, img_train_num, img_test_num, img_val_num))
    print('去除无法下载的图像，数据集共有 %d 个图像，训练集大小是 %d，测试集大小是 %d，验证集大小是 %d' % (img_valid_num, img_train_valid_num, img_test_valid_num, img_val_valid_num))
    print('SIND数据集共有 %d 个图像，%d个annotation，图像的annotation分布是：' % (img_num, anno_num))
    print('annotation数量, 图像数量，比例')
    for term in img_stat:
        print '%-3d%-8d%-5.3f' % (term,img_stat[term],float(img_stat[term])/img_num)
    print('训练集有 %d 个图像，%d个annotation，图像的annotation分布是：' % (img_train_num,anno_train_num))
    print('annotation数量, 图像数量，比例')
    for term in img_train_stat:
        print '%-3d%-8d%-5.3f' % (term,img_train_stat[term],float(img_train_stat[term])/img_train_num)
    print('测试集有 %d 个图像，%d个annotation，图像的annotation分布是：' % (img_test_num,anno_test_num))
    print('annotation数量, 图像数量，比例')
    for term in img_test_stat:
        print '%-3d%-8d%-5.3f' % (term,img_test_stat[term],float(img_test_stat[term])/img_test_num)
    print('验证集有 %d 个图像，%d个annotation，图像的annotation分布是：' % (img_val_num,anno_val_num))
    print('annotation数量, 图像数量，比例')
    for term in img_val_stat:
        print '%-3d%-8d%-5.3f' % (term,img_val_stat[term],float(img_val_stat[term])/img_val_num)


def anno_repeated(story_set):

    anno_num = {}
    for term in story_set:
        img_set = set(story_set[term]['image_id'])
        length = len(img_set)
        if anno_num.has_key(length):
            anno_num[length] = anno_num[length]+1
        else:
            anno_num[length]=1
    print '每个story包含5个不同的图片，没有相同的图片'
    print anno_num


def main(params):

    sis_test = json.load(open(params['input_sis_test'], 'r'))
    sis_train = json.load(open(params['input_sis_train'], 'r'))
    sis_val = json.load(open(params['input_sis_val'], 'r'))
    print 'json keys:'
    print sis_test.keys()
    print len(sis_test['albums'])
    print len(sis_test['annotations'])
    print len(sis_test['images'])
    print json.dumps(sis_test['albums'][0], indent=2)
    print json.dumps(sis_test['annotations'][0], indent=2)
    print json.dumps(sis_test['images'][0], indent=2)

    image_set = get_image_set(sis_test, sis_train, sis_val)
    story_set = get_story_set(sis_test, sis_train, sis_val, image_set)
    story_set_refine = refine_story_set(story_set)
    print('---------------------------------------------------------------------------------')
    print('原始SIND数据集的图像和SIS的分布情况：')
    calclt_img(image_set, story_set)
    print('---------------------------------------------------------------------------------')
    print('SIND数据集中的部分图像无法下载，导致对应的SIS的标注信息就需要删除')
    print('原始SIND数据集的图像和处理后的SIS的分布情况：')
    calclt_img(image_set, story_set_refine)
    anno_repeated(story_set_refine)
    json.dump(story_set_refine, open(params['output_json'], 'w'))
    json.dump(image_set, open('image_set.json', 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', default='', help='root file folder of test/train/val images')
    parser.add_argument('--input_sis_test', default='SIS/test.story-in-sequence.json', help='input sis test json file')
    parser.add_argument('--input_sis_train', default='SIS/train.story-in-sequence.json', help='input sis train json file')
    parser.add_argument('--input_sis_val', default='SIS/val.story-in-sequence.json', help='input sis val json file')
    parser.add_argument('--output_json', default='story_set.json', help='output the preprocessed sis data json file')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent=2)
    main(params)