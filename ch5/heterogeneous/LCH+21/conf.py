
conf = {
	"data_type" : "image",
	"model_name" : "mlp",
	"no-iid": "",
	"global_epochs" : 1000,
	"local_epochs" : 3,
	"beta" : 0.5,
	"batch_size" : 64,
	"weight_decay":1e-5,
	"lr" : 0.001,
	"momentum" : 0.9,
	"num_classes": 2,
	"num_parties":10,
	"is_init_avg": True,
	"split_ratio": 0.3,
	"label_column": "label",
	"data_column": "file",
	"test_dataset": "./data/cifar10/test/test.csv",
	"train_dataset" : "./data/cifar10/train/train.csv",
	"model_dir":"./save_model/",
	"model_file":"model.pth",
	"retrain":{
		"epoch": 10,
		"lr": 0.001,
		"num_vr":2000
	}
}