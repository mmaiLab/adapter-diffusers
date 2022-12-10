import timm
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from copy import deepcopy
from dataclasses import dataclass

ROOT = Path('/gs/hs1/tga-i/otake.s.ad/diffusion')
MODELS = ROOT / 'emotion_model'
idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear',
                4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}

class_weights = {
    0: 74874,
    1: 134415,
    2: 25459,
    3: 14090,
    4: 6378,
    5: 3803,
    6: 24882,
    7: 3750,
}


def map_idx_to_class(x):
    return idx_to_class[x]


def ret_emo(tensor):
    return list(map(map_idx_to_class, tensor.tolist()))


@dataclass
class EmotionOutput:
    feature: torch.tensor
    pred_proba: torch.tensor
    emotion: list


class AffectNetDataset(Dataset):
    def __init__(self, data_type='train'):
        super().__init__()
        annot_file = ROOT / 'AffectNet' / \
            f'{data_type}_set' / 'annotations.csv'
        self.data_dir = annot_file.parent / 'images'
        self.df = pd.read_csv(annot_file)
        self.df['id'] = self.df['id'].astype(int)
        self.df['emotion'] = self.df['emotion'].astype(int)

        IMG_SIZE = 224
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )

    def get_weight(self):
        self.emotion_labels = ['Neutral', 'Happiness', 'Sadness',
                               'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
        self.class_to_idx = {}
        self.idx_to_class = {}
        for i, emotion in enumerate(self.emotion_labels):
            self.class_to_idx[emotion] = i
            self.idx_to_class[i] = emotion
        sample_label, sample_counts = np.unique(
            self.df['emotion'].values, return_counts=True)
        for l, c in zip(sample_label, sample_counts):
            print(f'{self.emotion_labels[l]}: {c} ', end='')
        print('')

        cw = 1/sample_counts
        cw /= cw.min()
        class_weights = {i: cwi for i, cwi in zip(sample_label, cw)}
        print(class_weights)
        return class_weights

    def _get_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        annots = self.df.loc[index]
        image_idx = int(annots['id'])
        emotion = int(annots["emotion"])
        valence = annots["valence"]
        arosal = annots["arosal"]

        image_path = self.data_dir / f'{image_idx}.jpg'
        image_tensor = self.image_transforms(self._get_image(image_path))

        item = {
            'images': image_tensor,
            'emotion': emotion,
            'valence': valence,
            'arosal': arosal,
        }
        return item


def collate_fn(batch):
    images = []
    emotions = []
    valences = []
    arosals = []
    item = {}
    for data in batch:
        images.append([data['images']])
        emotions.append([data['emotion']])
        valences.append([data['valence']])
        arosals.append([data['arosal']])
    images = torch.stack(images)
    emotions = torch.stack(emotions)
    arosals = torch.stack(arosals)
    valences = torch.stack(valences)
    item = {
        'images': images,
        'emotions': emotions,
        'arosals': arosals,
        'valences': valences,
    }
    return item


class EmotionRecognitionModel(nn.Module):
    def __init__(self,
                 model_name='tf_efficientnet_b0_ns',
                 freeze_param=True,
                 pretrained_model='state_vggface2_enet0_new.pt',
                 classifier=None,
                 preprocess=False
                 ):
        super().__init__()

        self.idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear',
                             4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}

        base_model = timm.create_model(model_name, pretrained=False)
        if classifier:
            base_model.classifier = classifier
        if preprocess:
            self.preprocessor = nn.Conv2d(4, 3, 1, bias=False)
        else:
            self.preprocessor = None

        if pretrained_model:
            base_model.load_state_dict(torch.load(
                MODELS / pretrained_model), strict=False)

        self.feature_extractor = deepcopy(base_model)
        self.feature_extractor.classifier = nn.Identity()
        self.classifier = deepcopy(base_model.classifier)
        if freeze_param:
            self._freeze_module(self.feature_extractor)
            self._freeze_module(self.classifier)

    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _map_idx_to_class(self, x):
        return self.idx_to_class[x]

    def ret_emo(self, tensor):
        return list(map(self._map_idx_to_class, tensor.tolist()))

    def forward(self, x, preprocess=False, emotion_map=False):
        if self.preprocessor and preprocess:
            x = self.preprocessor(x)

        if self.feature_extractor:
            feature = self.feature_extractor(x)
            output = self.classifier(feature)
        else:
            output = self.model(x)
            feature = None

        if emotion_map:
            emotion = ret_emo(output.argmax(dim=-1))
        else:
            emotion = output.argmax(dim=-1)

        return EmotionOutput(
            feature=feature,
            pred_proba=output,
            emotion=emotion,
        )


# loss function
class AffectNetCriterion(nn.Module):
    def __init__(self, device, weight, label_smooth=False, multitask=False):
        super().__init__()
        self.weights = torch.FloatTensor([*weight.values()]).to(device)
        if label_smooth:
            self.loss_emotions = self.cross_entropy_with_label_smoothing
        else:
            self.loss_emotions = nn.CrossEntropyLoss(weight=self.weights)
        self.multitask = multitask
        if multitask:
            self.loss_valence = nn.MSELoss()  # nn.MSELoss()
            self.loss_arousal = nn.MSELoss()  # nn.MSELoss()

    def forward(self, preds, targets):

        num_classes = preds.shape[1]

        if self.multitask:
            # targets = [emotions, valences, arosals]
            loss_emotions = self.loss_emotions(
                preds[:, :num_classes], targets[0]
            )
            loss_valence = self.loss_valence(
                preds[:, num_classes], targets[1]
            )
            loss_arousal = self.loss_arousal(
                preds[:, num_classes + 1], targets[2]
            )
            output = loss_emotions + (loss_valence + loss_arousal) * 1
        else:
            loss_emotions = self.loss_emotions(preds[:, :num_classes], targets)
            output = loss_emotions
        return output

    def label_smooth(self, target, n_classes, label_smoothing=0.1):
        # convert to one-hot
        batch_size = target.size(0)
        target = torch.unsqueeze(target, 1)
        soft_target = torch.zeros(
            (batch_size, n_classes), device=target.device)
        soft_target.scatter_(1, target, 1)
        # label smoothing
        soft_target = soft_target * \
            (1 - label_smoothing) + label_smoothing / n_classes
        return soft_target

    def cross_entropy_loss_with_soft_target(self, pred, soft_target):
        #logsoftmax = nn.LogSoftmax(dim=-1)
        return torch.mean(torch.sum(- self.weights * soft_target * torch.nn.functional.log_softmax(pred, -1), 1))

    def cross_entropy_with_label_smoothing(self, pred, target):
        soft_target = self.label_smooth(target, pred.size(1))  # num_classes) #
        return self.cross_entropy_loss_with_soft_target(pred, soft_target)


# def get_criterion()


# criterion = cross_entropy_with_label_smoothing
