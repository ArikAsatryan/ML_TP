import numpy as np

data = np.loadtxt('HW2_labels.txt',  delimiter=',')
y_predict, y_true = data[:, :2], data[:, -1]


def accuracy_score(y_true, y_predict, percent=None):
    percent = 50 if percent == None else percent
    if percent > 100 or percent < 0:
        return -1
    
    threshold = percent/100
    count = (int)(y_predict.shape[0] * threshold)    
    predict = y_predict[:count].argmax(axis = 1)

    
    #TP = if predict == 1 and y_true == 1
    TP = np.sum(np.array([predict[i] == 1 and y_true[i] == 1 for i in range(predict.shape[0])]))
    
    #TN = if predict == 0 and y_true == 0
    TN = np.sum(np.array([predict[i] == 0 and y_true[i] == 0 for i in range(predict.shape[0])]))
    
    #FP = if predict == 1 and y_true == 0
    FP = np.sum(np.array([predict[i] == 1 and y_true[i] == 0 for i in range(predict.shape[0])]))
    
    #FN = if predict == 0 and y_true == 1
    FN = np.sum(np.array([predict[i] == 0 and y_true[i] == 1 for i in range(predict.shape[0])]))
    
    result = (TP + TN)/(TP + TN + FP + FN)
    return result


def precision_score(y_true, y_predict, percent=None):
    percent = 50 if percent == None else percent
    if percent > 100 or percent < 0:
        return -1
    
    threshold = percent/100
    count = (int)(y_predict.shape[0] * threshold)    
    predict = y_predict[:count].argmax(axis = 1)

    TP = np.sum(np.array([predict[i] == 1 and y_true[i] == 1 for i in range(predict.shape[0])]))
    TN = np.sum(np.array([predict[i] == 0 and y_true[i] == 0 for i in range(predict.shape[0])]))
    FP = np.sum(np.array([predict[i] == 1 and y_true[i] == 0 for i in range(predict.shape[0])]))
    FN = np.sum(np.array([predict[i] == 0 and y_true[i] == 1 for i in range(predict.shape[0])]))
    return TP / (TP + FP)


def recall_score(y_true, y_predict, percent=None):
    percent = 50 if percent == None else percent
    if percent > 100 or percent < 0:
        return -1
    
    threshold = percent/100
    count = (int)(y_predict.shape[0] * threshold)    
    predict = y_predict[:count].argmax(axis = 1)

    TP = np.sum(np.array([predict[i] == 1 and y_true[i] == 1 for i in range(predict.shape[0])]))
    TN = np.sum(np.array([predict[i] == 0 and y_true[i] == 0 for i in range(predict.shape[0])]))
    FP = np.sum(np.array([predict[i] == 1 and y_true[i] == 0 for i in range(predict.shape[0])]))
    FN = np.sum(np.array([predict[i] == 0 and y_true[i] == 1 for i in range(predict.shape[0])]))
    return TP / (TP + FN)


def f1_score(y_true, y_predict, percent=None):
    percent = 50 if percent == None else percent
    if percent > 100 or percent < 0:
        return -1  
    
    recall = recall_score(y_true, y_predict, percent)
    precision = precision_score(y_true, y_predict, percent)
    return 2*(precision * recall)/(precision + recall)


def lift_score(y_true, y_predict, percent=None):
    percent = 50 if percent == None else percent
    if percent > 100 or percent < 0:
        return -1
    
    threshold = percent/100
    count = (int)(y_predict.shape[0] * threshold)    
    predict = y_predict[:count].argmax(axis = 1)

    precision = precision_score(y_true, y_predict, percent)
    TP = np.sum(np.array([predict[i] == 1 and y_true[i] == 1 for i in range(predict.shape[0])]))
    FN = np.sum(np.array([predict[i] == 0 and y_true[i] == 1 for i in range(predict.shape[0])]))
    return precision * predict.shape[0]  / (TP + FN)

