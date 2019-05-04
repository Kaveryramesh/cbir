# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import infer
from DB import Database

import glob
from color import Color
from daisy import Daisy
from edge  import Edge
from gabor import Gabor
from HOG   import HOG
from vggnet import VGGNetFeat
from resnet import ResNetFeat
import scipy.misc
import operator

depth = 1
d_type = 'd1'
# query_idx = 125

def inc(map, key):
  if key not in map.keys():
    map[key] = 1
  else:
    map[key] +=  1

def test(db, query_idx):
  results = {}

  # retrieve by color
  method = Color()
  samples = method.make_samples(db)
  query = samples[query_idx]
  # print(samples)
  img = scipy.misc.imread(query['img'])
  # print(query)
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  # results.append(result[0]['cls'])
  inc(results, result[0]['cls'])

  # # retrieve by daisy
  # method = Daisy()
  # samples = method.make_samples(db)
  # query = samples[query_idx]
  # _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  # # results.append(result[0]['cls'])
  # inc(results, result[0]['cls'])

  # # # retrieve by edge
  method = Edge()
  samples = method.make_samples(db)
  query = samples[query_idx]
  # print(samples)
  query = samples[query_idx]
  img = scipy.misc.imread(query['img'])
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  # results.append(result[0]['cls'])
  inc(results, result[0]['cls'])

  # # # retrieve by gabor
  # # method = Gabor()
  # # samples = method.make_samples(db)
  # # query = samples[query_idx]
  # # _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  # # print(result)
  # # inc(results, result[0]['cls'])

  # # retrieve by HOG
  method = HOG()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  # results.append(result[0]['cls'])
  inc(results, result[0]['cls'])

  # # retrieve by VGG
  method = VGGNetFeat()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  # results.append(result[0]['cls'])
  inc(results, result[0]['cls'])

  # # retrieve by resnet
  method = ResNetFeat()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  # results.append(result[0]['cls'])
  inc(results, result[0]['cls'])

  import os
  from PIL import Image
  print(results)
  finalresult=max(results.items(), key=operator.itemgetter(1))[0]
  #string=".../database/"+finalresult+"/"
  string="./database/"+finalresult+"/"
  print(string)
  a=1
  for file in os.listdir(string):
    a+=1
    tempimg=Image.open(string+file)
    tempimg.show()

    print(string+file)
    if(a==10):
      break
  print("Final result is: ", finalresult)

  scipy.misc.imshow(img)

if __name__ == '__main__':
  db = Database()
  db.load_test()

  for i in range(9):
    test(db, i)

  # test(db, 1)
