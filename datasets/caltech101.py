import os

from .utils import Datum, DatasetBase
from .oxford_pets import OxfordPets


template = ['a photo of a {}.']
negative_template = ['a photo without {}.']


class Caltech101(DatasetBase):

    dataset_dir = 'caltech-101'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '101_ObjectCategories')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Caltech101.json')

        self.template = template
        self.negative_template = negative_template
        self.cupl_path = './gpt3_prompts/CuPL_prompts_caltech101.json'

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)