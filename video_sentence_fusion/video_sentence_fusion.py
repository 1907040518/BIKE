import torch
import sys
sys.path.append('.')
from utils.utils import accuracy
torch.set_printoptions(profile="full")

def fusion_acc():
    video_labels_list = torch.load('video_sentence_fusion/hmdb51_video_labels.pt')
    video_sim = torch.load('video_sentence_fusion/hmdb51_video_sims.pt')

    sentence_labels_list = torch.load('video_sentence_fusion/hmdb51_sentence_labels.pt')
    sentence_sim = torch.load('video_sentence_fusion/hmdb51_sentence_sims.pt')

    print('video_labels==', video_labels_list.shape)
    print('sentence_label===', sentence_labels_list.shape)
    print((video_labels_list == sentence_labels_list).sum())

    a = 0
    b = 1
    
    fusion_matrix = a * video_sim + b * sentence_sim
    fusion_prec = accuracy(fusion_matrix, video_labels_list, topk=(1, 5))
    print('a=={}'.format(a), 'b=={}'.format(b), 'top1=={}'.format(fusion_prec[0].item()), 'top5=={}'.format(fusion_prec[1].item()))

fusion_acc()
