import time
import torch
import data_loader
import numpy as np

from scipy import stats
from models import ScoreNet


class Solver(object):
    """Solver for training and testing hyperIQA"""

    def __init__(self, config, train_idx, test_idx, train_test_num):

        self.epochs = config.epochs
        self.save_model = config.dataset + "_" + str(test_idx[0]) + "_" + str(test_idx[1]) + "_round" + str(train_test_num + 1) + ".pth"
        self.tr_te_num = train_test_num
        self.d_set = config.dataset
        train_loader = data_loader.DataLoader(config.dataset, train_idx, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, test_idx, batch_size=1, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

        self.model = ScoreNet().cuda()
        self.l1_loss = torch.nn.L1Loss().cuda()
        self.lr_1 = config.lr_1
        self.lr_2 = config.lr_2
        self.weight_decay = config.weight_decay

        resnet_params = list(map(id, self.model.RES.parameters()))
        self.other_params = filter(lambda p: id(p) not in resnet_params, self.model.parameters())

        paras = [{'params': self.other_params, 'lr': self.lr_1},
                 {'params': self.model.RES.parameters(), 'lr': self.lr_2}
                 ]
        self.optimizer = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        torch.optim.lr_scheduler.StepLR(self.optimizer, 20, gamma=0.9, last_epoch=-1)

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        print('Epoch\t\tTrain_Loss\t\tTrain_SRCC\t\tTrain_PLCC\t\tTest_SRCC\t\tTest_PLCC')
        for t in range(self.epochs):
            self.model.train()
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            one_training_epoch_start_time = time.time()
            for i, (img, label) in enumerate(self.train_data):
                if i % 200 == 0:
                    print("Trian: current batsize: %d, total batchsize: %d" % (i, len(self.train_data)))
                img = torch.as_tensor(img.cuda())
                label = torch.as_tensor(label.cuda())

                self.optimizer.zero_grad()

                pred = self.model(img)
                for p_lst in pred.cpu().tolist():
                    pred_scores.append(p_lst[0])
                for g_lst in label.cpu().tolist():
                    gt_scores.append(g_lst[0])
                loss = self.l1_loss(pred, label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            one_training_epoch_end_time = time.time()
            if t == 0:
                print("one_training_epoch_time: %4.3fs" % (one_training_epoch_end_time - one_training_epoch_start_time))
            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            train_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                torch.save(self.model.state_dict(), self.save_model)
                if best_srcc > 0.87:
                    torch.save(self.model.state_dict(), self.save_model[:-4] + "_SROCC%4.3f" % best_srcc + ".pth")

            print('%d\t\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, train_plcc, test_srcc, test_plcc))

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model.eval()
        pred_scores = []
        gt_scores = []

        one_testing_epoch_start_time = time.time()
        with torch.no_grad():
            for _, (img, label) in enumerate(data):
                img = img.squeeze(0)
                img = img.cuda()
                label = label.squeeze(0)
                label = label.cuda()

                pred = self.model(img)
                pred_score = pred.mean()
                pred_score = pred_score.item()
                gt_score = label.item()
                pred_scores.append(pred_score)
                gt_scores.append(gt_score)

        one_testing_epoch_end_time = time.time()
        print("one_testing_epoch_time: %4.3fs" % (one_testing_epoch_end_time - one_testing_epoch_start_time))
        pred_scores = np.array(pred_scores)
        gt_scores = np.array(gt_scores)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        return test_srcc, test_plcc
