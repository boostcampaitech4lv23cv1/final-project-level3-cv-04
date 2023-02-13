from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings
import os

from ..dataset import ImageDataset


class KPOP(ImageDataset):
    dataset_dir = 'kpop'

    def __init__(self, root='/opt/ml/torchkpop/body_embedding/data', **kwargs):
        '''
        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        '''
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir) # '/opt/ml/torchkpop/body_embedding/data/kpop'
        
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        
        
        
        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)
        
        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        
        # print("### train")
        # print(train)
        # print("### query")
        # print(query)
        # print("### gallery")
        # print(gallery)
        
        super(KPOP, self).__init__(train, query, gallery, **kwargs)
        
    def get_pid_parser(self, dir_path):
        '''
            parser['VhHicXLaDos_aespa_karina'] = pid
        '''
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        
        parse_unique = set()
        for img_path in img_paths:
            filename = os.path.basename(img_path)
            # ex) filename = "VhHicXLaDos_0_40_{trackID}_{dfIndex}_{groupname}_{pred1}"
            tags = filename.split('_')
            linkID = tags[0] # VhHicXLaDos
            group = tags[5] # aespa
            member = tags[6] # karina
            
            parse = linkID + '_' + group + '_' + member # 'VhHicXLaDos_aespa_karina'
            
            parse_unique.add(parse)
            
        parser = dict()
        for i, parse in enumerate(sorted(parse_unique)):
            parser[parse] = i
        return parser
    
    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        pid_parser = self.get_pid_parser(dir_path)
        
        data = []
        for img_path in img_paths:
            filename = os.path.basename(img_path)
            # ex) filename = VhHicXLaDos_0_40_{trackID}_{dfIndex}_{groupname}_{pred1}
            tags = filename.split('_')
            linkID = tags[0] # VhHicXLaDos
            start_sec = tags[1] # 0
            end_sec = tags[2] # 40
            track_id = tags[3]
            df1_index = tags[4]
            group = tags[5] # aespa
            member = tags[6] # karina
            
            parse = linkID + '_' + group + '_' + member # 'VhHicXLaDos_aespa_karina'
            pid = pid_parser[parse]
            
            data.append((img_path, pid, 0))
            # data.append({'img_path': img_path,
            #              'pid': pid,
            #              'camid': 0})
        return data

    def get_member_num(self):
        return len(self.meta_info['member_list'])