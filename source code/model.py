#!/usr/bin/python2.7
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import os
from batch_gen import moving_average
from eval import evaluate

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes_verb, num_classes_noun, num_classes_action):
        super(MultiStageModel, self).__init__()

        self.stage1_verb = SingleStageModel(num_layers, num_f_maps, dim, num_classes_verb)
        self.stage1_noun = SingleStageModel(num_layers, num_f_maps, dim, num_classes_noun)

        self.stages_verb = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes_verb+num_classes_noun, num_classes_verb)) for s in range(num_stages-1)])
        self.stages_noun = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes_noun+num_classes_verb, num_classes_noun)) for s in range(num_stages-1)])

        self.action_conv1 = ConvNet(num_f_maps, num_classes_action)
        self.action_convs = nn.ModuleList([copy.deepcopy(ConvNet(num_f_maps, num_classes_action)) for s in range(num_stages-1)])

    def forward(self, x, mask_verb, mask_noun):
        input = x
        out_mid_verb, out_fi_verb = self.stage1_verb(input, mask_verb)
        out_mid_noun, out_fi_noun = self.stage1_noun(input, mask_noun)
        outputs_action = self.action_conv1(out_mid_verb, out_mid_noun).unsqueeze(0)
        outputs_verb, outputs_noun = out_fi_verb.unsqueeze(0), out_fi_noun.unsqueeze(0)

        for s_verb, s_noun, action_conv in zip(self.stages_verb, self.stages_noun, self.action_convs):

            out_mid_verb, out_fi_verb = s_verb(torch.cat([F.softmax(out_fi_verb, dim=1), F.softmax(out_fi_noun, dim=1)], dim=1) * mask_verb[:, 0:1, :], mask_verb)
            out_mid_noun, out_fi_noun = s_noun(torch.cat([F.softmax(out_fi_noun, dim=1), F.softmax(out_fi_verb, dim=1)], dim=1) * mask_noun[:, 0:1, :], mask_noun)
            output_action = action_conv(out_mid_verb, out_mid_noun)
            outputs_verb, outputs_noun = torch.cat((outputs_verb, out_fi_verb.unsqueeze(0)), dim=0), torch.cat((outputs_noun, out_fi_noun.unsqueeze(0)), dim=0)
            outputs_action = torch.cat((outputs_action, output_action.unsqueeze(0)), dim=0)

        return outputs_verb, outputs_noun, outputs_action


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out_mid = self.conv_1x1(x)
        for layer in self.layers:
            out_mid = layer(out_mid, mask)
        out_fi = self.conv_out(out_mid) * mask[:, 0:1, :]
        return out_mid, out_fi


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class ConvNet(nn.Module):
    def __init__(self, num_f_maps, num_classes):
        super(ConvNet, self).__init__()
        self.dilation = 7
        self.conv_gs_kernels = 7
        self.dilated_convs = nn.ModuleList([copy.deepcopy(nn.Conv1d(num_f_maps, num_classes, 1)) for _ in range(self.conv_gs_kernels)])
        param_bound = np.sqrt(1/num_f_maps/self.conv_gs_kernels)
        for param in self.dilated_convs.parameters():
            nn.init.uniform_(param, a=-param_bound, b=param_bound)
        self.conv_gs = nn.Conv1d(num_f_maps, 1, 3, padding='same')

    def forward(self, input1, input2):

        # x = torch.cat((input1, input2), dim=1)
        x = input1 + input2
        gs = self.conv_gs(x)
        sigma = 1/(1+torch.exp(-gs))
        frac_sigma = 1/sigma
        x_len = x.shape[2]

        outputs = 0
        for i, dilated_conv in enumerate(self.dilated_convs):
            mid = int(len(self.dilated_convs) / 2)
            shift = int(abs(i - mid))
            if len(self.dilated_convs) % 2 == 0 and i >= mid:
                shift += 1

            distance = self.dilation*shift/x_len
            bili = torch.exp(-0.5 * distance ** 2 * frac_sigma ** 2)

            x_ = F.pad(x, (shift * self.dilation, shift * self.dilation), 'constant', 0)
            if i < mid:
                output = dilated_conv(x_[:, :, :-2*shift*self.dilation])*bili
            elif i >= mid:
                output = dilated_conv(x_[:, :, 2*shift*self.dilation:])*bili

            outputs += output

        return outputs


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes_verb, num_classes_noun, num_classes_action, args):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes_verb, num_classes_noun, num_classes_action)
        self.mse = nn.MSELoss(reduction='none')
        self.args = args
        self.num_classes_verb, self.num_classes_noun, self.num_classes_action = num_classes_verb, num_classes_noun, num_classes_action
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.smooth_f = 0.15

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)

        optimizer = optim.Adam([*self.model.parameters()], lr=learning_rate)

        for epoch in range(num_epochs):

            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target_verb, batch_target_noun, batch_target_action, mask_verb, mask_noun, mask_action = batch_gen.next_batch(batch_size)
                batch_input, batch_target_verb, batch_target_noun, batch_target_action, mask_verb, mask_noun, mask_action = \
                    batch_input.to(device), batch_target_verb.to(device), batch_target_noun.to(device), batch_target_action.to(device), mask_verb.to(device), mask_noun.to(device), mask_action.to(device)

                optimizer.zero_grad()
                
                predictions_verb, predictions_noun, predictions_action = self.model(batch_input, mask_action, mask_noun)

                loss = 0
                i = 0
                for p_verb, p_noun, p_action in zip(predictions_verb, predictions_noun, predictions_action):

                    loss_cross_verb = self.ce(p_verb.transpose(2, 1).contiguous().view(-1, self.num_classes_verb), batch_target_verb.view(-1))
                    loss += loss_cross_verb
                    mse_smooth_verb = torch.clamp(self.mse(F.log_softmax(p_verb[:, :, 1:], dim=1),
                                                                     F.log_softmax(p_verb.detach()[:, :, :-1], dim=1)), min=0, max=16)
                    loss += self.smooth_f * torch.mean(mse_smooth_verb * mask_verb[:, :, 1:])

                    loss_cross_noun = self.ce(p_noun.transpose(2, 1).contiguous().view(-1, self.num_classes_noun), batch_target_noun.view(-1))
                    loss += loss_cross_noun
                    mse_smooth_noun = torch.clamp(self.mse(F.log_softmax(p_noun[:, :, 1:], dim=1),
                                                                     F.log_softmax(p_noun.detach()[:, :, :-1], dim=1)), min=0, max=16)
                    loss += self.smooth_f * torch.mean(mse_smooth_noun * mask_noun[:, :, 1:])

                    loss_cross_action = self.ce(p_action.transpose(2, 1).contiguous().view(-1, self.num_classes_action), batch_target_action.view(-1))
                    loss += loss_cross_action
                    mse_smooth_action = torch.clamp(self.mse(F.log_softmax(p_action[:, :, 1:], dim=1),
                                                         F.log_softmax(p_action.detach()[:, :, :-1], dim=1)), min=0, max=16)
                    loss += self.smooth_f * torch.mean(mse_smooth_action * mask_action[:, :, 1:])

                    i += 1

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions_action[-1].data, 1)
                correct += ((predicted == batch_target_action).float()).sum().item()
                total += torch.sum(mask_action[:, 0, :]).item()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path1, features_path2, vid_list_file, epochs, actions_dict, device, sample_rate, gt_path, split):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            max_epoch = epochs
            max_f1 = 0.0
            print("*" * 10 + split + "*" * 10)
            for epoch in range(1, max_epoch + 1):
                self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
                file_ptr = open(vid_list_file, 'r')
                list_of_vids = file_ptr.read().split('\n')[:-1]
                file_ptr.close()

                for vid in list_of_vids:
                    features1 = np.load(os.path.join(features_path1, vid.split('.')[0] + '.npy'))  # 2048 934
                    features2 = np.load(os.path.join(features_path2, vid.split('.')[0] + '.npy'))  # 2048 934
                    min_len = min(features1.shape[1], features2.shape[1])
                    features = np.concatenate((features1[:, :min_len], features2[:, :min_len]), axis=0)

                    features = features[:, ::sample_rate]
                    input_x = torch.tensor(features, dtype=torch.float)
                    input_x.unsqueeze_(0)
                    input_x = input_x.to(device)

                    feature_verb, feature_noun = input_x[:, :, :], input_x[:, :, :]
                    mask_verb, mask_noun = torch.ones(feature_verb.size(), device=device), torch.ones(feature_noun.size(), device=device)
                    _, _, predictions = self.model(input_x, mask_verb, mask_noun)
                    _, predicted = torch.max(predictions[-1].data, 1)
                    predicted = predicted.squeeze()
                    recognition = []
                    for i in range(len(predicted)):
                        recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                        list(actions_dict.values()).index(
                                                                            predicted[i].item())]] * sample_rate))
                    f_name = vid.split('/')[-1].split('.')[0]
                    f_ptr = open(results_dir + "/" + f_name, "w")
                    f_ptr.write("### Frame level recognition: ###\n")
                    f_ptr.write(' '.join(recognition))
                    f_ptr.close()

                results = evaluate(gt_path, results_dir + "/", list_of_vids)
                if results[2] > max_f1:
                    max_f1 = results[2]
                    print("better epoch:{:d}".format(epoch))
                    print(results)
