import json
from conf import conf
import torch
import numpy as np
from fedavg.server import Server
from fedavg.client import Client
from fedavg.models import CNN_Model,weights_init_normal, ReTrainModel,MLP
from utils import get_data
import copy
from ot_preprocess import local_barycenter, global_barycenter, OTProjector
import torchvision.transforms.functional as TF
from PIL import Image
import os
from data_process import process_cifar10

def compute_global_latent_space(train_datasets):
    local_bcs = []
    for agent_id, df in train_datasets.items():
        imgs = []
        for path in df[conf["data_column"]].values:
            img = Image.open(path).convert("RGB")
            tensor_img = TF.to_tensor(img).numpy() * 255  # shape: (3, H, W)
            imgs.append(tensor_img)
        imgs = np.stack(imgs)
        bc = local_barycenter(imgs)
        local_bcs.append(bc)
    return global_barycenter(local_bcs)

if __name__ == '__main__':

    # PROCESSES CIFAR-10 DATASET BEFORE CODE RUNS
    if not os.path.exists('./data/cifar10/train/train.csv') or not os.path.exists('./data/cifar10/test/test.csv'):
        print("CIFAR-10 data not found. Processing and saving...")
        process_cifar10('./data', './data/cifar10')
    else:
        print("CIFAR-10 image data found. Skipping preprocessing.")

    train_datasets, val_datasets, test_dataset = get_data()

    print("Computing OT global latent space...")
    global_bc = compute_global_latent_space(train_datasets)
    projector = OTProjector(global_bc)
    print("Projection module ready.")

    client_weight = {}
    if conf["is_init_avg"]:
        for key in train_datasets.keys():
            client_weight[key] = 1 / len(train_datasets)


    clients = {}
    clients_models = {}

    if conf['model_name'] == "mlp":
        n_input = test_dataset.shape[1] - 1
        model = MLP(n_input, 512, conf["num_classes"])
    elif conf['model_name'] == 'cnn':
        model = CNN_Model()
    model.apply(weights_init_normal)

    if torch.cuda.is_available():
        model.cuda()


    server = Server(conf, model, test_dataset, projector=projector)


    for key in train_datasets.keys():
        clients[key] = Client(conf, server.global_model, train_datasets[key], val_datasets[key], projector=projector)


    if not os.path.isdir(conf["model_dir"]):
        os.mkdir(conf["model_dir"])
    max_acc = 0

    for e in range(conf["global_epochs"]):

        for key in clients.keys():
            print('training client {}...'.format(key))
            model_k = clients[key].local_train(server.global_model)
            clients_models[key] = copy.deepcopy(model_k)

        server.model_aggregate(clients_models, client_weight)
        acc, loss = server.model_eval()
        print("Epoch %d, global_acc: %f, global_loss: %f\n" % (e, acc, loss))

    if conf['no-iid'] == 'fed_ccvr':
        client_mean = {}
        client_cov = {}
        client_length = {}

        for key in clients.keys():
            c_mean, c_cov, c_length = clients[key].cal_distributions(server.global_model)
            client_mean[key] = c_mean
            client_cov[key] = c_cov
            client_length[key] = c_length
        g_mean, g_cov = server.cal_global_gd(client_mean, client_cov, client_length)

        retrain_vr = []
        label = []
        eval_vr = []
        for i in range(conf['num_classes']):
            mean = np.squeeze(np.array(g_mean[i]))
            vr = np.random.multivariate_normal(mean, g_cov[i], conf["retrain"]["num_vr"]*2)
            retrain_vr.extend(vr.tolist()[:conf["retrain"]["num_vr"]])
            eval_vr.extend(vr.tolist()[conf["retrain"]["num_vr"]:])
            label.extend([i]*conf["retrain"]["num_vr"])

        retrain_model = ReTrainModel()
        if torch.cuda.is_available():
            retrain_model.cuda()
        reset_name = []
        for name, _ in retrain_model.state_dict().items():
            reset_name.append(name)

        for name, param in server.global_model.state_dict().items():
            if name in reset_name:
                retrain_model.state_dict()[name].copy_(param.clone())

        retrain_model = server.retrain_vr(retrain_vr, label, eval_vr, retrain_model)

        for name, param in retrain_model.state_dict().items():
            server.global_model.state_dict()[name].copy_(param.clone())

        acc, loss = server.model_eval()
        print("After retraining global_acc: %f, global_loss: %f\n" % (acc, loss))


    torch.save(server.global_model.state_dict(), os.path.join(conf["model_dir"],conf["model_file"]))
