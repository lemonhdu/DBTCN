import torch
import os
import argparse
import random
from model_cp import Trainer
from batch_gen import BatchGenerator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')
parser.add_argument('--epoch', default="100")
args = parser.parse_args()

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 4096
bz = 1
lr = 0.0005
num_epochs = int(args.epoch)

sample_rate = 1
if args.dataset == "50salads":
    sample_rate = 2  

abs_dir_head = r'/disks/disk0/huangxvfeng/dataset/segmentation/'
vid_list_file = abs_dir_head + args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = abs_dir_head + args.dataset+"/splits/test.split"+args.split+".bundle"

features_path_3D = abs_dir_head + args.dataset + "/features_3d_2048/"
features_path_2D = abs_dir_head + args.dataset + "/local_2d_features_valid_ce/split"+str(args.split)

gt_path_verb = abs_dir_head + args.dataset + "/groundTruth_grain_85_verb/"
gt_path_noun = abs_dir_head + args.dataset + "/groundTruth_grain_85_noun/"
gt_path_action = abs_dir_head + args.dataset + "/groundTruth_grain_85_action/"

mapping_file_verb = abs_dir_head + args.dataset + "/mapping_85_verb.txt"
mapping_file_noun = abs_dir_head + args.dataset + "/mapping_85_noun.txt"
mapping_file_action = abs_dir_head + args.dataset + "/mapping_85_action.txt"

model_dir = "./models/"+args.dataset+"/split_"+args.split+"/"
results_dir = "./results/"+args.dataset+"/split_"+args.split+"/"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# verb
file_ptr = open(mapping_file_verb, 'r')
verbs = file_ptr.read().split('\n')[:-1]
file_ptr.close()
verbs_dict = dict()
for a in verbs:
    verbs_dict[a.split(' ')[1]] = int(a.split(' ')[0])
num_classes_verb = len(verbs)

# noun
file_ptr = open(mapping_file_noun, 'r')
nouns = file_ptr.read().split('\n')[:-1]
file_ptr.close()
noun_dict = dict()
for a in nouns:
    noun_dict[a.split(' ')[1]] = int(a.split(' ')[0])
num_classes_noun = len(nouns)

# action
file_ptr = open(mapping_file_action, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
action_dict = dict()
for a in actions:
    action_dict[a.split(' ')[1]] = int(a.split(' ')[0])
num_classes_action = len(actions)


trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes_verb, num_classes_noun, num_classes_action, args)

if args.action == "train":
    batch_gen = BatchGenerator(num_classes_verb, num_classes_noun, num_classes_action, verbs_dict, noun_dict, action_dict,
                               gt_path_verb, gt_path_noun, gt_path_action, features_path_3D, features_path_2D, sample_rate)

    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path_3D, features_path_2D, vid_list_file_tst, num_epochs, action_dict, device,
                    sample_rate, gt_path_action, args.split)
