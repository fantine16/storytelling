from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import json

annFile = './annotations/captions_val2014.json'
resFile = './results/captions_val2014_fakecap_results.json'

f1 = json.load(open(annFile, 'r'))
f2 = json.load(open(resFile, 'r'))

coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

cocoEval = COCOEvalCap(coco, cocoRes)
cocoEval.params['image_id'] = cocoRes.getImgIds()
cocoEval.evaluate()
# print output evaluation scores
for metric, score in cocoEval.eval.items():
    print '%s: %.3f'%(metric, score)

print('finish')