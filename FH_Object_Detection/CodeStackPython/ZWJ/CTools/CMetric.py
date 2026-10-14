
class CMetric():

    def calculate_metrics(PRED, GT, ignore_label_list):
        assert len(PRED) == len(GT), "预测标签和真值标签的数量必须相等"

        ignore_label_nums = len(ignore_label_list)
        num_classes = max(max(PRED), max(GT)) + 1 - ignore_label_nums
        # num_classes = 8
        confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
        TP = [0] * num_classes
        TN = [0] * num_classes
        FP = [0] * num_classes
        FN = [0] * num_classes

        for pred, gt in zip(PRED, GT):
            for i in range(num_classes):
                if pred == i and gt == i:  # TP 预测为正，标签为正，预测正确T TP
                    TP[i] += 1
                elif pred != i and gt != i:  # TN 预测为负，标签为负，预测正确T TN
                    TN[i] += 1
                elif pred == i and gt != i:  # FN 预测为正，标签为负，预测错误F FN
                    # FP[i] += 1
                    FN[i] += 1
                elif pred != i and gt == i:  # FP 预测为负，标签为正，预测错误F FP
                    # FN[i] += 1
                    FP[i] += 1
            confusion_matrix[gt][pred] += 1

        accuracy = sum(TP) / len(PRED)
        class_iou = []
        mean_iou = 0.0

        for i in range(num_classes):
            union = TP[i] + FP[i] + FN[i]
            if union == 0:
                iou = 0.0
            else:
                iou = TP[i] / union
            class_iou.append(iou)
            mean_iou += iou

        mean_iou /= num_classes

        return TP, TN, FP, FN, accuracy, class_iou, mean_iou

