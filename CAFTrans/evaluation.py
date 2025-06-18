from datasets.own_data import ImageFolder
from torch.utils.data import DataLoader
import logging
import numpy as np
__all__ = ['SegmentationMetric']
import torch
import os
import cv2
from logger import get_looger


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
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def Precision(self):
        #tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        #precision = tp / (tp + fp)
        tp = np.diag(self.confusionMatrix)
        fp = np.sum(self.confusionMatrix, axis=1) - np.diag(self.confusionMatrix)
        #fn = np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        precision = tp / (tp + fp)
        precision = np.nanmean(precision)
        return precision

    def Recall(self):
        #tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        tp = np.diag(self.confusionMatrix)
        #fp = np.sum(self.confusionMatrix, axis=1) - np.diag(self.confusionMatrix)
        fn = np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        recall = tp / (tp + fn)
        recall = np.nanmean(recall)
        return recall

    def F1(self):
        #tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        tp = np.diag(self.confusionMatrix)
        fp = np.sum(self.confusionMatrix, axis=1) - np.diag(self.confusionMatrix)
        fn = np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        recall_1 = tp / (tp + fn)
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


def write_img(pred_images, filename, ori_image):
    pred = pred_images[0]  # pred的shape为（高，宽，类别数量）
    pred = pred.permute(1,2,0)
    pred = pred.cpu().numpy()
    COLORMAP = [[0, 0, 0], [0, 0, 255], [0, 255, 0]]  # 分别为0-3类对应的颜色
    cm = np.array(COLORMAP).astype(np.uint8)

    pred = np.argmax(np.array(pred), axis=2)  # 此时pred的shape为（高，宽）
    pred_val = cm[pred]  # pred_val的shape为（高，宽，3）

    # 将模型预测结果叠加在原图上
    ori_image = ori_image[0].permute(1,2,0)
    ori_image = ori_image.cpu().numpy()
    img_save = pred_val*0.6+ori_image*0.4
    #if ori_image.size() == pred_val.size():
    #overlap = cv2.addWeighted(pred_val, 0.7, pred_val, 0.3, 0)
    cv2.imwrite(filename, img_save)
    #cv2.imwrite(filename, pred_val)
    #cv2.imwrite(os.path.join("gdrive", "My Drive", "data", "deeplab", filename.split("/")[-1]), overlap)



def test_evaluation(args, model):
    db_test = ImageFolder(args, split='test')
    testloader = DataLoader(dataset=db_test, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    total_num = int(len(testloader))
    precision = np.zeros((1, total_num))
    recall = np.zeros((1, total_num))
    oa = np.zeros((1, total_num))
    F1 = np.zeros((1, total_num))
    mIoU = np.zeros((1, total_num))
    # logger = get_looger(args.save_log)
    # logger.info('start training')

    for i_batch, sample in enumerate(testloader):
        n = i_batch
        img_mask = sample['label']
        img_input = sample['image'].cuda()
        img_mask = img_mask.squeeze(0).cpu().detach().numpy()
        model.eval()

        with torch.no_grad():
            img_prec = model(img_input)
            img_prec = torch.argmax(torch.sigmoid(img_prec), dim=1).squeeze(0)
            img_prec = img_prec.cpu().detach().numpy()
            img_mask=np.asarray(img_mask,dtype=int)
            imgPredict = img_prec  # 可直接换成预测图片
            imgLabel = img_mask  # 可直接换成标注图片
        #x,y = img_prec.shape
        #pa_sum = 0
        #for i in range(x):
        #    for j in range(y):
        #        if imgPredict[i][j] == imgLabel[i][j]:
        #            pa_sum += 1
        #pa1 = pa_sum/(x*y)
        #print(pa1)
            metric = SegmentationMetric(3)  # 3表示有3个分类，有几个分类就填几
            metric.addBatch(imgPredict, imgLabel)
            oa_ = metric.OA()
            mIoU_ = metric.meanIntersectionOverUnion()
            precision_ = metric.Precision()
            recall_ = metric.Recall()
            F1_ = metric.F1()
        precision[0, n] = precision_  # precision = tp / (tp + fp) + 1e-6
        recall[0, n] = recall_  # recall = tp / (tp + fn)
        F1[0, n] = F1_  # (2 * pre[0, n] * rec[0, n]) / (pre[0, n] + rec[0, n])
        oa[0, n] = oa_  # (tp + tn) / (tp + fp + tn + fn)
        mIoU[0, n] = mIoU_  # tp / (tp + fp + fn + 1e-6)

    mean_oa = np.mean(oa)
    mean_mIoU = np.mean(mIoU)
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_F1 = np.mean(F1)
        #pa = metric.pixelAccuracy()
        #print(pa)
        #FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
        #mpa = metric.meanPixelAccuracy()
        #mIoU = metric.meanIntersectionOverUnion()
        #print("********************")
        #print('pa is : %f' % pa)
        #print('cpa is :')  # 列表
        #print('FWIoU is : %f' % FWIoU)
        #print('mpa is : %f' % mpa)
        #print('mIoU is : %f' % mIoU)
        #print("********************")

    print("********************")
    print("********************")
    print('mean value')
    print('OA_mean=', np.mean(oa))
    print('mIOU_mean=', np.mean(mIoU))
    print('precision_mean=', np.mean(precision))
    print('recall_mean=', np.mean(recall))
    print('F1_mean=', np.mean(F1))
    print("********************")

    # logger.info('finish testing!')


def save_image_mask(args, model):
    db_test = ImageFolder(args, split='test')
    testloader = DataLoader(dataset=db_test, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logging.info("{} test iterations per epoch".format(len(testloader)))

    for i_batch, sample in enumerate(testloader):

        img_input = sample['image'].cuda()
        model.eval()

        with torch.no_grad():
            img_prec = model(img_input)
            save_path = os.path.join(args.save_image_path, args.model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_image_path = save_path + '/%s.png' % sample['case_name'][0]
            write_img(img_prec, save_image_path, img_input)
            img_prec = torch.argmax(torch.sigmoid(img_prec), dim=1).squeeze(0)
            img_prec = img_prec.cpu().detach().numpy()
            img_prec[img_prec > 0] = 255
            img_prec = np.array(img_prec, dtype=np.float32)


            if os.path.exists(save_image_path):
                print('image was saved')
                continue
            else:
                cv2.imwrite(save_image_path, img_prec, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                print('%s save image' % sample['case_name'][0])
                continue

    print('finish save')

