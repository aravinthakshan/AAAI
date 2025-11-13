import os
import torch


class Config:
    def __init__(self):
        dataset_dir = '/kaggle/input/camodata'
        self.dp = DataPath(dataset_dir)
        self.num_workers = 4 # changed this for kaggle

        self.CUDA = True
        self.device = torch.device('cuda' if self.CUDA else 'cpu')

        self.epochs = 200 # changed this for testing WandB
        self.trainsize = 384
        self.batch_size = 8
        self.weight_decay = 4e-8
        self.learning_rate = 2.6e-4
        self.min_lr = 1e-7


# class DataPath:
#     def __init__(self, dataset_dir):
#         self.dataset_dir = dataset_dir

#         ''' Train Dataset: CAMO-Train + COD10K-Train '''
#         self.train_imgs = os.path.join(self.dataset_dir, 'TrainDataset', 'Imgs')
#         self.train_masks = os.path.join(self.dataset_dir, 'TrainDataset', 'GT')

#         ''' Test Dataset '''
#         # CHAMELEON
#         self.test_CHAMELEON_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'CHAMELEON', 'Imgs')
#         self.test_CHAMELEON_masks = os.path.join(self.dataset_dir, 'TestDataset', 'CHAMELEON', 'GT')
#         # CAMO-Test
#         self.test_CAMO_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'CAMO', 'Imgs')
#         self.test_CAMO_masks = os.path.join(self.dataset_dir, 'TestDataset', 'CAMO', 'GT')
#         # COD10K-Test
#         self.test_COD10K_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'COD10K', 'Imgs')
#         self.test_COD10K_masks = os.path.join(self.dataset_dir, 'TestDataset', 'COD10K', 'GT')
#         # NC4K
#         self.test_NC4K_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'NC4K', 'Imgs')
#         self.test_NC4K_masks = os.path.join(self.dataset_dir, 'TestDataset', 'NC4K', 'GT')

class DataPath:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        ''' Train Dataset: CAMO-Train + COD10K-Train '''
        self.train_imgs = os.path.join(self.dataset_dir, 'COD-TrainDataset', 'COD-TrainDataset', 'Imgs')
        self.train_masks = os.path.join(self.dataset_dir, 'COD-TrainDataset', 'COD-TrainDataset', 'GT')

        ''' Test Dataset '''
        # CHAMELEON
        self.test_CHAMELEON_imgs = os.path.join(self.dataset_dir, 'COD-TestDataset', 'COD-TestDataset', 'CHAMELEON', 'Imgs')
        self.test_CHAMELEON_masks = os.path.join(self.dataset_dir, 'COD-TestDataset', 'COD-TestDataset', 'CHAMELEON', 'GT')
        # CAMO-Test
        self.test_CAMO_imgs = os.path.join(self.dataset_dir, 'COD-TestDataset', 'COD-TestDataset', 'CAMO', 'Imgs')
        self.test_CAMO_masks = os.path.join(self.dataset_dir, 'COD-TestDataset', 'COD-TestDataset', 'CAMO', 'GT')
        # COD10K-Test
        self.test_COD10K_imgs = os.path.join(self.dataset_dir, 'COD-TestDataset', 'COD-TestDataset', 'COD10K', 'Imgs')
        self.test_COD10K_masks = os.path.join(self.dataset_dir, 'COD-TestDataset', 'COD-TestDataset', 'COD10K', 'GT')
        # NC4K
        self.test_NC4K_imgs = os.path.join(self.dataset_dir, 'COD-TestDataset', 'COD-TestDataset', 'NC4K', 'Imgs')
        self.test_NC4K_masks = os.path.join(self.dataset_dir, 'COD-TestDataset', 'COD-TestDataset', 'NC4K', 'GT')
