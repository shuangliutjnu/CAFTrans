import torch
import numpy as np
__all__ = ['SegmentationMetric']
from logger import get_looger
from datasets.own_data import ImageFolder
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from tqdm import tqdm
import logging
import sys
import os
from datetime import datetime


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def OA(self):
                # return all class overall pixel accuracy
                #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
                # return each category pixel accuracy(A more accurate way to call it precision)
                # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        #classAcc = np.nan_to_num(classAcc)
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
                # Intersection = TP Union = TP + FP + FN
                # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        # IoU = intersection / (union + 1e-6)  # 返回列表，其值为各个类别的IoU
        IoU = intersection / (union)
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def Precision(self):
        #tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        #precision = tp / (tp + fp)
        tp = np.diag(self.confusionMatrix)
        fp = np.sum(self.confusionMatrix, axis=1) - np.diag(self.confusionMatrix)
        fn = np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        # precision = tp / (tp + fp + 1e-6)
        precision = tp / (tp + fp)
        precision = np.nanmean(precision)
        return precision

    def Recall(self):
        #tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        tp = np.diag(self.confusionMatrix)
        fp = np.sum(self.confusionMatrix, axis=1) - np.diag(self.confusionMatrix)
        fn = np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        # recall = tp / (tp + fn + 1e-6)
        recall = tp / (tp + fn)
        recall = np.nanmean(recall)
        return recall

    def F1(self):
        #tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        tp = np.diag(self.confusionMatrix)
        fp = np.sum(self.confusionMatrix, axis=1) - np.diag(self.confusionMatrix)
        fn = np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        # recall_1 = tp / (tp + fn + 1e-6)
        # precision_1 = tp / (tp + fp + 1e-6)
        recall_1 = tp / (tp + fn )
        precision_1 = tp / (tp + fp)
        Recall = np.nanmean(recall_1)
        Precision = np.nanmean(precision_1)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
                # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
                # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


def trainer_dataset(args, model):
    logging.basicConfig(filename=args.save_log + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    batch_size = args.batch_size #args.n_gpu *
    lr = args.lr
    db_train = ImageFolder(args=args, split='train')   #ImageFolder
    db_test = ImageFolder(args=args, split='test')
    train_loader = DataLoader(dataset=db_train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


    ce_loss = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))

    best_oa = 0
    best_iou = 0
    #max_iterations = args.max_epoch * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    #log.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    lr1 = lr*0.1

    for epoch in iterator:
        now1 = datetime.now()
        model.train()
        loss_sum = 0.0
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(non_blocking=True), label_batch.cuda(non_blocking=True)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss = loss_ce
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_total = len(train_loader)
            iter_num += 1
            lr_ = lr * (1.0 - iter_num / max_iterations) ** 0.9
            if epoch >= 15:
                lr_ = lr1 * (1.0 - iter_num / max_iterations) ** 0.9  # lr1 = lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            #logging.info('epoch %d  iteration %d  /  %d : loss : %f, ' % (epoch, (iter_num - iter_total * epoch), iter_total, loss.item(),))
        logging.info('epoch %d loss is %f' % (epoch, loss_sum / iter_total))
        now2 = datetime.now()
        print('this model time is:' , now2 - now1)
        if epoch >= 0:
            test_loader = DataLoader(dataset=db_test, shuffle=False, num_workers=0, pin_memory=True)
            test_number = len(test_loader)
            logging.info('%d  images is testing' % test_number)
            precision = np.zeros((1, test_number))
            recall = np.zeros((1, test_number))
            oa = np.zeros((1, test_number))
            F1 = np.zeros((1, test_number))
            mIoU = np.zeros((1, test_number))
            #pre = np.zeros((1, test_number))
            #accuracy = np.zeros((1, test_number))
            for n, sample in enumerate(test_loader):
                img_mask = sample['label'].cuda(non_blocking=True)
                img_input = sample['image'].cuda(non_blocking=True)
                img_mask = img_mask.squeeze(0).cpu().detach().numpy()
                model.eval()
                with torch.no_grad():
                    img_prec = model(img_input)
                    img_prec = torch.argmax(torch.sigmoid(img_prec), dim=1).squeeze(0)
                    img_prec = img_prec.cpu().detach().numpy()

                    img_mask = np.asarray(img_mask, dtype=int)
                    imgPredict = img_prec  # 可直接换成预测图片
                    imgLabel = img_mask  # 可直接换成标注图片
                    metric = SegmentationMetric(3)  # 3表示有3个分类，有几个分类就填几
                    metric.addBatch(imgPredict, imgLabel)
                    oa_ = metric.OA()
                    mIoU_ = metric.meanIntersectionOverUnion()
                    precision_ = metric.Precision()
                    recall_ = metric.Recall()
                    F1_ = metric.F1()
                    #pa = metric.pixelAccuracy()
                    #FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
                    #mpa = metric.meanPixelAccuracy()
                    #mIoU = metric.meanIntersectionOverUnion()
                precision[0, n] = precision_  # precision = tp / (tp + fp) + 1e-6
                recall[0, n] = recall_  # recall = tp / (tp + fn)
                F1[0, n] = F1_  # (2 * pre[0, n] * rec[0, n]) / (pre[0, n] + rec[0, n])
                oa[0, n] = oa_  # (tp + tn) / (tp + fp + tn + fn)
                mIoU[0, n] = mIoU_
            mean_oa = np.mean(oa)
            mean_mIoU = np.mean(mIoU)
            mean_precision = np.mean(precision)
            mean_recall = np.mean(recall)
            mean_F1 = np.mean(F1)
            logging.info(
                'Epoch:[{}/{}]\t loss={:.5f}\t oa={:.5f}\t mIoU=={:.5f}\t precision=={:.5f}\t recall=={:.5f}\t F1={:.5f}'.format(
                    epoch, args.max_epoch, loss, mean_oa, mean_mIoU, mean_precision, mean_recall, mean_F1))
            save_model_path = os.path.join(args.save_path, args.model_name)
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            if mean_mIoU > best_iou:
                logging.info('new best mIoU %f is saved' % mean_mIoU)
                save_pre_path = os.path.join(save_model_path, 'best_iou' + 'epoch' + str(epoch) + '.pth')

                # save total model or dict
                if args.model_saved_mode == 'total model':
                    torch.save(model, save_pre_path, _use_new_zipfile_serialization=False)
                else:
                    torch.save(model.state_dict(), save_pre_path, _use_new_zipfile_serialization=False)
                best_iou = mean_mIoU

            if mean_oa > best_oa:
                logging.info('new best oa %f is saved' % mean_oa)
                save_acc_path = os.path.join(save_model_path, 'best_oa' + 'epoch' + str(epoch) + '.pth')

                # save total model or dict
                if args.model_saved_mode == 'total model':
                    torch.save(model, save_acc_path, _use_new_zipfile_serialization=False)
                else:
                    torch.save(model.state_dict(), save_acc_path, _use_new_zipfile_serialization=False)
                best_oa = mean_oa

            if epoch % 5 == 0 :
                save_epoch_path = os.path.join(save_model_path, 'epoch' + str(epoch) + '.pth')
                if args.model_saved_mode == 'total model':
                    torch.save(model, save_epoch_path, _use_new_zipfile_serialization=False)
                else:
                    torch.save(model.state_dict(), save_epoch_path, _use_new_zipfile_serialization=False)


    return "Training Finished!"













