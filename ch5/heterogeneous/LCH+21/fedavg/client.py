import torch
import numpy as np
from fedavg.datasets import get_dataset

class Client(object):

    def __init__(self, conf, model, train_df, val_df, projector=None):
        self.conf = conf
        self.projector = projector

        self.local_model = model
        self.train_df = train_df
        self.train_dataset = get_dataset(conf, self.train_df)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],shuffle=True)

        self.val_df = val_df
        self.val_dataset = get_dataset(conf, self.val_df)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=conf["batch_size"],shuffle=True)

    def local_train(self, model):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'],weight_decay=self.conf["weight_decay"])
        criterion = torch.nn.CrossEntropyLoss()
        for e in range(self.conf["local_epochs"]):
            self.local_model.train()
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                feature, output = self.local_model(data)

                loss = criterion(output, target)
                loss.backward()

                optimizer.step()
            train_acc, train_loss = self.model_eval(loader=self.train_loader)
            val_acc, val_loss = self.model_eval(loader=self.val_loader)
            acc, eval_loss = self.model_eval()
            print("Epoch {0} done. train_loss ={1}, eval_loss = {2}, eval_acc={3}".format(e, loss, eval_loss, acc))
            print(
                f"Epoch {e} done. Train Acc={train_acc:.2f}%, Train Loss={train_loss:.4f}, " 
                f"Val Acc={val_acc:.2f}%, Val Loss={val_loss:.4f}"
            )

        return self.local_model.state_dict()

    @torch.no_grad()
    def model_eval(self, loader=None):
        self.local_model.eval()

        if loader is None:
            loader = self.val_loader

        total_loss = 0.0
        correct = 0
        dataset_size = 0

        criterion = torch.nn.CrossEntropyLoss()

        for batch_id, batch in enumerate(loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            _, output = self.local_model(data)

            total_loss += criterion(output, target)
            pred = output.data.max(1)[1]  
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss.cpu().detach().numpy() / dataset_size

        return acc, total_l

    def _cal_mean_cov(self,features):
        features = np.array(features)
        mean = np.mean(features, axis=0)

        cov = np.cov(features.T, bias=1)
        return mean,cov

    def cal_distributions(self, model):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        self.local_model.eval()

        features = []
        mean = []
        cov = []
        length = []

        for i in range(self.conf["num_classes"]):
            train_i = self.train_df[self.train_df[self.conf['label_column']] == i]
            train_i_dataset = get_dataset(self.conf, train_i)

            if len(train_i_dataset) > 0:
                train_i_loader = torch.utils.data.DataLoader(train_i_dataset, batch_size=self.conf["batch_size"],
                                                             shuffle=True)
                for batch_id, batch in enumerate(train_i_loader):
                    data, target = batch

                    if torch.cuda.is_available():
                        data = data.cuda()

                    feature, _ = self.local_model(data)
                    features.extend(feature.tolist())

                f_mean, f_cov = self._cal_mean_cov(features)

            else:
                f_mean = np.zeros((256,))
                f_cov = np.zeros((256,256))

            mean.append(f_mean)
            cov.append(f_cov)
            length.append(len(train_i))

        return mean, cov, length







