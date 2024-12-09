import torchmetrics
import torch

class MulticlassMetrics():
    def __init__(self, device, num_classes):
        self.device = device
        self.used_keys = {}
        self.num_classes = num_classes
        self.init_metrics()

    def init_metrics(self):
        # Initialize metrics for multiclass classification
        self.classifier_metrics_dict = {
            "acc": torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average=None).to(self.device),
            # "kappa": torchmetrics.CohenKappa(task='multiclass', num_classes=self.num_classes).to(self.device),
            "prec": torchmetrics.Precision(task='multiclass', num_classes=self.num_classes, average=None).to(self.device),
            "recall": torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average=None).to(self.device),
            "f1": torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average=None).to(self.device),
        }

    def fill_metrics(self, raw_predictions, raw_labels):
        # Convert raw predictions to probabilities and get predicted classes
        predictions = torch.softmax(raw_predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1)
        labels = torch.argmax(raw_labels, dim=1)
        # breakpoint()

        # Update metrics
        self.classifier_metrics_dict["acc"].update(predictions, labels)
        # self.classifier_metrics_dict["kappa"].update(predictions, labels)
        self.classifier_metrics_dict["prec"].update(predictions, labels)
        self.classifier_metrics_dict["recall"].update(predictions, labels)
        self.classifier_metrics_dict["f1"].update(predictions, labels)

        self.used_keys = {
            "acc": True,
            "prec": True,
            "recall": True,
            "f1": True,
        }

    def compute_and_log_metrics(self, loss=0):
        metrics = {}
        for item in self.used_keys:
            metrics[item] = self.classifier_metrics_dict[item].compute()

        if loss != 0:
            metrics["loss_cross_entropy"] = loss

        return metrics

    def clear_metrics(self):
        for _, metric in self.classifier_metrics_dict.items():
            metric.reset()
        self.used_keys = {}


if __name__ == "__main__":
    y = torch.tensor([[0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 0., 1.]])
    
    y_pred = torch.tensor([[ 0.0183,  0.0983,  0.0312],
        [ 0.0353,  0.0725,  0.0136],
        [ 0.0222,  0.0433,  0.0121],
        [-0.0441,  0.3260,  0.1144],
        [-0.1817,  0.5961,  0.1830],
        [ 0.0222,  0.0433,  0.0121],
        [ 0.0190,  0.1135,  0.0246],
        [ 0.0222,  0.0433,  0.0121],
        [ 0.0162,  0.0622,  0.0173],
        [ 0.0263,  0.0617,  0.0220]])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    metrics = MulticlassMetrics(device, num_classes)

    metrics.fill_metrics(y_pred, y)