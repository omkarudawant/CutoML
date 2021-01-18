from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def calculate_metrics(y_true, y_pred):
    f1 = f1_score(
        y_true=y_true, y_pred=y_pred, average="weighted") * 100
    precision = precision_score(
        y_true=y_true, y_pred=y_pred, average="weighted") * 100
    recall = recall_score(
        y_true=y_true, y_pred=y_pred, average="weighted") * 100
    accuracy = accuracy_score(
        y_true=y_true, y_pred=y_pred) * 100

    final_scores = f" Accuracy: {accuracy} | F1: {f1} | Precision: " \
                   f"{precision} | Recall: {recall}"
    print(final_scores)

    return accuracy, f1, precision, recall,


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        "Time elapsed: {:0>2}:{:0>2}:{:05.2f} hh:mm:ss".format(
            int(hours),
            int(minutes),
            seconds)
    )
