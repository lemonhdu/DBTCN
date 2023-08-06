#!/usr/bin/python2.7
import torch
import numpy as np
import random
import os

def moving_average(x, w):

    outs = []

    for dim in range(len(x)):
        out = np.convolve(x[dim], np.ones(w), 'same')/w
        out[:w], out[len(x[dim])-w:] = x[dim][:w], x[dim][len(x[dim])-w:]
        out = out.reshape(1, len(x[dim]))
        outs.append(out)

    out = np.concatenate([i for i in outs], axis=0)

    return out


class BatchGenerator(object):
    def __init__(self, num_classes_verb, num_classes_noun, num_classes_action, verbs_dict, noun_dict, action_dict,
                 gt_path_verb, gt_path_noun, gt_path_action, features_path_3D, features_path_2D, sample_rate):

        self.list_of_examples = list()
        self.index = 0
        self.num_classes_verb, self.num_classes_noun, self.num_classes_action = num_classes_verb, num_classes_noun, num_classes_action
        self.verbs_dict, self.noun_dict, self.action_dict = verbs_dict, noun_dict, action_dict
        self.gt_path_verb, self.gt_path_noun, self.gt_path_action = gt_path_verb, gt_path_noun, gt_path_action
        self.features_path_3D = features_path_3D
        self.features_path_2D = features_path_2D
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target_verb = []
        batch_target_noun = []
        batch_target_action = []
        for vid in batch:
            features_3D = np.load(os.path.join(self.features_path_3D, vid.split('.')[0] + '.npy'))  # 2048 934
            features_2D = np.load(os.path.join(self.features_path_2D, vid.split('.')[0] + '.npy'))  # 2048 934

            min_len = min(features_3D.shape[1], features_2D.shape[1])

            features = np.concatenate((features_3D[:, :min_len], features_2D[:, :min_len]), axis=0)

            # verb label
            file_ptr = open(self.gt_path_verb + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.verbs_dict[content[i]]
            batch_target_verb.append(classes[::self.sample_rate])

            # noun label
            file_ptr = open(self.gt_path_noun + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.noun_dict[content[i]]
            batch_target_noun.append(classes[::self.sample_rate])

            # action label
            file_ptr = open(self.gt_path_action + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.action_dict[content[i]]
            batch_target_action.append(classes[::self.sample_rate])

            # temporal alignment
            if np.shape(features)[1] > len(content):
                dec = (np.shape(features)[1]-len(content))//2
                features = features[:, dec:len(content)+dec]
            batch_input.append(features[:, ::self.sample_rate])

        length_of_sequences_verb, length_of_sequences_noun, length_of_sequences_action = \
            list(map(len, batch_target_verb)), list(map(len, batch_target_noun)), list(map(len, batch_target_action))

        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences_action), dtype=torch.float)

        batch_target_tensor_verb = torch.ones(len(batch_input), max(length_of_sequences_verb), dtype=torch.long)*(-100)
        batch_target_tensor_noun = torch.ones(len(batch_input), max(length_of_sequences_noun), dtype=torch.long) * (-100)
        batch_target_tensor_action = torch.ones(len(batch_input), max(length_of_sequences_action), dtype=torch.long) * (-100)

        mask_verb = torch.zeros(len(batch_input), self.num_classes_verb, max(length_of_sequences_verb),
                                  dtype=torch.float)
        mask_noun = torch.zeros(len(batch_input), self.num_classes_noun, max(length_of_sequences_noun),
                                  dtype=torch.float)
        mask_action = torch.zeros(len(batch_input), self.num_classes_action, max(length_of_sequences_action),
                                  dtype=torch.float)

        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor_verb[i, :np.shape(batch_target_verb[i])[0]] = torch.from_numpy(batch_target_verb[i])
            batch_target_tensor_noun[i, :np.shape(batch_target_noun[i])[0]] = torch.from_numpy(batch_target_noun[i])
            batch_target_tensor_action[i, :np.shape(batch_target_action[i])[0]] = torch.from_numpy(batch_target_action[i])

            mask_verb[i, :, :np.shape(batch_target_verb[i])[0]] = torch.ones(self.num_classes_verb, np.shape(batch_target_verb[i])[0])
            mask_noun[i, :, :np.shape(batch_target_noun[i])[0]] = torch.ones(self.num_classes_noun, np.shape(batch_target_noun[i])[0])
            mask_action[i, :, :np.shape(batch_target_action[i])[0]] = torch.ones(self.num_classes_action, np.shape(batch_target_action[i])[0])

        return batch_input_tensor, batch_target_tensor_verb, batch_target_tensor_noun, batch_target_tensor_action, mask_verb, mask_noun, mask_action

