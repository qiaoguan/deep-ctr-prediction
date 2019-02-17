from collections import defaultdict
from sklearn.metrics import roc_auc_score
import numpy as np

'''
calculate group_auc and cross_entropy_loss(log loss for binary classification)

@author: Qiao
'''


def cal_group_auc(labels, preds, user_id_list):
    """Calculate group auc"""

    print('*' * 50)
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    #
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    return group_auc


def cross_entropy_loss(labels, preds):
    """calculate cross_entropy_loss

      loss = -labels*log(preds)-(1-labels)*log(1-preds)

      Args:
        labels, preds

      Returns:
         log loss
    """

    if len(labels) != len(preds):
        raise ValueError(
            "labels num should equal to the preds num,")

    z = np.array(labels)
    x = np.array(preds)
    res = -z * np.log(x) - (1 - z) * np.log(1 - x)
    return res.tolist()
