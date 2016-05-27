#-*-coding:utf-8-*-
import json
import argparse
import string
import os
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize
import cv2

def prepro_captions(story):
    print 'example processed tokens:'
    for i,term in enumerate(story):
        term['processed_tokens'] =[]
        for j, s in enumerate(term['text']):
            txt = str(s.encode('utf-8')).lower().translate(None, string.punctuation).strip().split()
            term['processed_tokens'].append(txt)
            if i < 10 and j == 0: print txt

def build_vocab(story,params):
    count_thr = params['word_count_threshold']
    # count up the number of words
    counts = {}
    for term in story:
        for txt in term['processed_tokens']:
            for w in txt:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str, cw[:20]))

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w, n in counts.iteritems() if n <= count_thr]
    vocab = [w for w, n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts))
    print 'number of words in vocab would be %d' % (len(vocab),)
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words)

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    story_lengths = {}
    for term in story:
        ns=0
        for txt in term['processed_tokens']:
            nw = len(txt)
            ns=ns+nw
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
            story_lengths[ns] = story_lengths.get(ns, 0) + 1
    max_len_sen = max(sent_lengths.keys())
    max_len_story=max(story_lengths.keys())
    print 'max length sentence in raw data: ', max_len_sen
    print 'sentence length distribution (count, number of words):'
    sum_len_sen = sum(sent_lengths.values())
    for i in xrange(max_len_sen + 1):
        print '%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len_sen)

    print 'max length story in raw data: ', max_len_story
    print 'story length distribution (count, number of words):'
    sum_len_story = sum(story_lengths.values())
    for i in xrange(max_len_story + 1):
        print '%2d: %10d   %f%%' % (i, story_lengths.get(i, 0), story_lengths.get(i, 0) * 100.0 / sum_len_story)

    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print 'inserting the special UNK token'
        vocab.append('UNK')
    for term in story:
        term['final_captions']=[]
        for txt in term['processed_tokens']:
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            term['final_captions'].append(caption)


    return vocab

def encode_captions(story, params, wtoi):
    max_length = params['max_length']
    N = len(story)
    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')
    counter = 1
    print ('story number %d' % len(story))
    for i,term in enumerate(story):
        #print i
        n = len(term['final_captions'])
        assert n > 0, 'error: some image has no captions'
        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(term['final_captions']):
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]
        label_arrays.append(Li)
        label_start_ix[i] = counter
        counter += n

    L = np.concatenate(label_arrays, axis=0)
    return L,label_start_ix

def main(params):

    story=json.load(open(params['story_set'],'r')).values() #list
    image_set=json.load(open(params['image_set'],'r')) #dict
    prepro_captions(story)
    vocab = build_vocab(story, params)

    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table
    N = len(story)*5
    L ,label_start_ix= encode_captions(story, params, wtoi)

    if not os.path.exists(params['output_h5']):
        f = h5py.File(params['output_h5'], "w")
        f.create_dataset("labels", dtype='uint32', data=L)
        f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
        dset = f.create_dataset("images", (N, 3, 256, 256), dtype='uint8')  # space for resized images
        num = 1
        for i, t in enumerate(story):

            for id in t['image_id']:
                num = num + 1
                imagename = image_set[id]['imagename'].encode('utf-8')
                split = image_set[id]['split'].encode('utf-8')
                if imagename.endswith('gif'):
                    I = imread('dataset/' + split + '/' + imagename)
                else:
                    I = cv2.imread('dataset/' + split + '/' + imagename)  # 这一步很慢，因为图片都比较大
                try:
                    Ir = imresize(I, (256, 256))
                except:
                    print 'failed resizing image %s ' % (split + '_used_valid/' + imagename)
                    raise
                if len(Ir.shape) == 2:
                    Ir = Ir[:, :, np.newaxis]
                    Ir = np.concatenate((Ir, Ir, Ir), axis=2)
                    # and swap order of axes from (256,256,3) to (3,256,256)
                Ir = Ir.transpose(2, 0, 1)
                # write to h5
                dset[i] = Ir
                if num % 10 == 0:
                    print 'processing %d/%d (%.2f%% done)' % (num, N, num * 100.0 / N)
        f.close()
        print 'wrote ', params['output_h5']
    else:
        print('%s 文件已经存在，请删除后再运行程序！'% params['output_h5'])
    out = {}
    out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['story']=story
    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_set', default='dataset/image_set.json', help='')
    parser.add_argument('--output_h5', default='dataset/storytelling.h5', help='output h5 file')
    parser.add_argument('--output_json', default='dataset/storytelling.json', help='output json file')

    parser.add_argument('--story_set', default='dataset/story_set.json', help='')

    parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print ('parsed input parameters:')
    print json.dumps(params, indent=2)
    main(params)