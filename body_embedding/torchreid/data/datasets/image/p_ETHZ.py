from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import glob
import warnings

from ..dataset import ImageDataset

# Sources :
# https://github.com/hh23333/PVPM
# A. Ess, B. Leibe, K. Schindler, and L. Van Gool, "A mobile vision system for robust multi-person tracking" in CVPR, 2008


class P_ETHZ(ImageDataset):
    dataset_dir = 'P_ETHZ'

    def __init__(self, root='', **kwargs):
        self.root=osp.abspath(osp.expanduser(root))
        # self.dataset_dir = self.root
        data_dir = osp.join(self.root, self.dataset_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated.')
        self.query_dir=osp.join(self.data_dir, 'occluded_body_images')
        self.gallery_dir=osp.join(self.data_dir, 'whole_body_images')

        train = []
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False, is_query=False)
        super(P_ETHZ, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False, is_query=True):
        img_paths = glob.glob(osp.join(dir_path, '*', '*.png'))
        if is_query:
            camid = 0
        else:
            camid = 1
        pid_container = set()
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            if relabel:
                pid = pid2label[pid]
            data.append({'img_path': img_path, 'pid': pid, 'camid': camid})
        return data

