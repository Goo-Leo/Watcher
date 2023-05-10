import numpy as np
from rknnlite.api import RKNNLite


model = ''

'''step1: Initialize'''
rknn_lite = RKNNLite()


'''step2: load RKNN model'''
print('--> Load RKNN model')
ret = rknn_lite.load_rknn(model)
if ret != 0:
    print('Load RKNN model failed')
    exit(ret)
print('done')

'''step3: Init runtime environment'''
print('--> Init runtime environment')
ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
if ret != 0:
    print('Init runtime environment failed')
    exit(ret)
print('done')


'''step4: Inference'''


'''step5: Release'''
rknn_lite.release()