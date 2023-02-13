from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
sys.path.append('/opt/ml/torchkpop/body_embedding')
from body_embedding.torchreid.data import ImageDataset
from body_embedding.torchreid.data import ImageDataManager
from body_embedding.torchreid.data import register_image_dataset
from body_embedding.torchreid.data.datasets.image import KPOP

from body_embedding.torchreid.utils import load_pretrained_weights
from body_embedding.torchreid.models import build_model
from body_embedding.torchreid.engine import ImageSoftmaxEngine
from body_embedding import torchreid
import glob

'''
    학습 완료된 가중치는 save_dir/model 에 저장됩니다.
'''

root_dir = '/opt/ml/torchkpop/body_embedding/data'

# load dataset
kpop_dataset = KPOP(mode='train')

# use your own dataset only
datamanager = ImageDataManager(
    root='/opt/ml/torchkpop/body_embedding/data', # ⭐️ 
    sources='kpop',
    height=800,
    width=400,
    batch_size_train=4,
    batch_size_test=4,
    transforms=["random_flip"]
)

def load_model():
    model = build_model( # 이거 나중에 문제될듯 지금은 나의 bpbreid 가상환경의 파이썬 실행 환경변수가 opt/ml/torchreid로 잘 되어 있어서 잘 되는 듯
        name="osnet_ain_x1_0",
        num_classes=kpop_dataset.num_train_pids, # 어차피 classifier 안써서 몇이든 상관없음
        loss="softmax",
        pretrained=False,
        use_gpu=True
    )
    load_weights = './pretrained_weight/osnet_ain_ms_d_c.pth.tar' # ✅ meta_info 를 이용해주세요
    load_pretrained_weights(model, load_weights)
    model.training = True
    return model

model = load_model()
model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim="adam",
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler="single_step",
    stepsize=20
)

engine = ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run( 
    save_dir="train_osnet/osnet_ain",
    max_epoch=100,
    start_epoch=0,
    print_freq=10,
    fixbase_epoch=20,
    eval_freq=10,
    test_only=False,
    dist_metric='euclidean',
    normalize_feature=False,
    visrank=False,
    visrank_topk=10,
    use_metric_cuhk03=False,
    ranks=[1, 5, 10, 20],
    rerank=False
)
