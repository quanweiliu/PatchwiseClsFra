# from MS2CANet.pymodel import pyCNN

# __all__ = ["pyCNN"]


def get_model_config(args):

	if args.backbone == "cnn" or args.backbone == "vit":
		args.epochs = 100
		args.batch_size = 64
		args.patch_size = 13
		args.randomCrop = 11
		args.pca = True
		args.components = 15
		# args.pca = False
		# args.components = 0
		args.learning_rate = 0.001
		##### original version no step_size and gamma
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.7

	elif args.backbone == "mamba":
		args.epochs = 100
		args.batch_size = 64
		args.patch_size = 12
		args.randomCrop = 8
		args.pca = True
		args.components = 15    
		# args.pca = False
		# args.components = 0
		args.learning_rate = 0.001
		##### original version no step_size and gamma
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.7

	elif args.backbone == "ViTDGCN":
		args.data_augmentation = False
		args.randomCrop = None
		args.batch_size = 128
		args.patch_size = 27
		args.pca = True
		args.components = 35
		# args.pca = False
		# args.components = 0
		args.learning_rate = 0.0005
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.5

	elif args.backbone == "FDGC":
		args.data_augmentation = False
		args.randomCrop = None
		args.batch_size = 128
		args.patch_size = 19
		args.pca = True
		args.components = 32
		# args.pca = False
		# args.components = 0
		args.learning_rate = 0.0001
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.5

	elif args.backbone == "SSFTTnet":
		args.data_augmentation = False
		args.randomCrop = None
		args.batch_size = 64
		args.patch_size = 13
		args.pca = True
		args.components = 30
		# args.pca = False      # NO
		# args.components = 0
		args.learning_rate = 0.001
		args.schedule = False
		# args.step_size = 30
		# args.gamma = 0.5

	elif args.backbone == "morphFormer":
		args.epochs = 500
		args.data_augmentation = False
		args.randomCrop = None
		args.batch_size = 64
		args.patch_size = 11
		args.pca = False
		args.components = 0
		args.learning_rate = 5e-4
		args.weight_decay = 5e-3
		args.FM = 16
		args.testSizeNumber = 100
		args.schedule = True
		args.step_size = 50
		args.gamma = 0.9

		# args.epochs = 100
		# args.data_augmentation = True
		# # args.randomCrop = 11
		# args.batch_size = 256
		# args.patch_size = 13
		# args.pca = False
		# args.components = 0
		# args.learning_rate = 1e-3
		# args.weight_decay = 1e-4
		# args.FM = 16
		# args.HSIOnly = True
		# args.testSizeNumber = 100
		# args.schedule = True
		# args.step_size = 30
		# args.gamma = 0.7

	elif args.backbone == "DBCTNet":
		# args.epochs = 300   # PU
		args.epochs = 500   # Houston
		# args.epochs = 600
		args.batch_size = 128
		args.patch_size = 9
		args.pca = False
		args.components = 0    
		# args.learning_rate = 0.001
		args.learning_rate = 0.003 # Houston
		args.weight_decay = 0.001
		# args.gammaF = 0.4  # PU
		args.gammaF = 1.7  # Houston

		##### original version no step_size and gamma
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.7

	elif args.backbone == "DSFormer":
		args.epochs = 500
		args.early_stop = "True"
		args.batch_size = 256
		args.patch_size = 10
		args.pca = True
		args.components = 30
		args.schedule = False
		args.learning_rate = 0.0001
		args.weight_decay = 0.00001
		args.ps = 2
		args.kernel_size = 3
		args.emb_dim = 128
		args.num_heads = 8
		args.group_num = 4
		args.k = '2/5'   # UP
		# args.k = '3/5'   # IP
		# args.k = '4/5'   # houston13 / whuhh 

	elif args.backbone == "MDL_M" or args.backbone == "MDL_L" or \
		args.backbone == "MDL_E_D" or args.backbone == "MDL_C":
		args.epochs = 150
		args.patch_size = 7
		args.batch_size = 64
		args.pca = False
		args.components = 0
		args.learning_rate = 0.001
		##### original version no step_size and gamma
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.5

	elif args.backbone == "FusAtNet":
		args.epochs = 1000
		args.patch_size = 11
		args.batch_size = 64
		args.pca = False
		args.components = 0
		args.learning_rate = 0.000005
		##### original version no step_size and gamma
		args.schedule = False
		# args.step_size = 30
		# args.gamma = 0.5

	elif  args.backbone == "CrossHL":
		args.epochs = 200
		args.patch_size = 11
		args.batch_size = 64
		args.pca = False
		args.components = 0
		args.learning_rate = 0.0005
		args.schedule = True
		args.step_size = 50
		args.gamma = 0.9

	elif args.backbone == "MS2CANet":
		args.epochs = 100
		args.patch_size = 11
		args.batch_size = 64
		args.learning_rate = 0.001
		args.pca = True
		args.components = 20    # Houston
		# args.pca = False
		# args.components = 0 
		##### original version no step_size and gamma
		args.schedule = False
		# args.step_size = 30
		# args.gamma = 0.5

	elif args.backbone == "S2ENet":
		args.epochs = 128
		args.batch_size = 64
		args.patch_size = 7
		args.pca = True
		args.components = 20    # Houston
		# args.pca = False
		# args.components = 0
		args.learning_rate = 0.001
		args.schedule = True

	elif args.backbone == "HCTNet":
		args.epochs = 100
		args.batch_size = 64
		args.patch_size = 11
		args.pca = True
		args.components = 20    # Houston
		# # args.components = 30    # Trento
		# args.pca = False      # NO
		# args.components = 0
		args.learning_rate = 0.001
		##### original version no step_size and gamma
		args.schedule = False
		# args.step_size = 30
		# args.gamma = 0.5

	elif args.backbone == "SHNet":
		args.epochs = 100
		args.batch_size = 128
		args.patch_size = 7
		args.pca = False
		args.components = 0
		args.factors = 4
		# args.factors = 8
		args.learning_rate = 0.001
		args.weight_decay = 0.001
		args.lb_smooth = 0.01
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.5

	elif args.backbone == "DSHFNet":
		args.epochs = 500
		args.batch_size = 64
		args.patch_size = 6
		args.pca = False
		args.components = 0
		args.learning_rate = 5e-4
		args.weight_decay = 0
		args.schedule = True
		args.step_size = args.epochs // 10
		args.gamma = 0.9

	elif args.backbone == "MIViT":
		args.epochs = 500
		args.patch_size = 8
		args.batch_size = 64
		args.distillation = 1
		args.pca = False
		args.components = 0
		args.learning_rate = 1e-4
		args.weight_decay = 0.001
		args.schedule = True
		args.step_size = args.epochs // 10
		args.gamma = 0.9
		args.fusion = 'TTOA'
		args.pred_flag = 'o_fuse'

	elif args.backbone == "EMamba":
		args.epochs = 300
		args.patch_size = 9
		args.batch_size = 128
		args.pca = False
		args.components = 0
		args.learning_rate = 0.0003
		args.weight_decay = 0.0001
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.5
		
	else:
		raise ValueError(f"Unsupported backbone: {args.backbone}")
	


from . import S2ENet, FusAtNet, SHNet, heads, MDL
# from models.MS2CANet import pymodel
from .MS2CANet2 import pymodel
from .CrossHL import CrossHL
from .HCTNet import HCTNet
from .DSHFNet import DSHF
from .MIViT import MMA
from .mamba.vmamba import MultimodalClassier


def get_model(data1_bands, data2_bands, class_num, args):
	if args.backbone == "MDL_M":
		model = MDL.Middle_fusion_CNN(data1_bands, data2_bands, class_num).to(args.device)
		params = model.parameters()
		print("model: ", "MDL_M")

	elif  args.backbone == "MDL_L":
		model = MDL.Late_fusion_CNN(data1_bands, data2_bands, class_num).to(args.device)
		params = model.parameters()
		print("model: ", "MDL_L")

	elif args.backbone == "MDL_E_D":
		model = MDL.En_De_fusion_CNN(data1_bands, data2_bands, class_num).to(args.device)
		params = model.parameters()
		print("model: ", "MDL_E_D")

	elif  args.backbone == "MDL_C":
		model = MDL.Cross_fusion_CNN(data1_bands, data2_bands, class_num).to(args.device)
		params = model.parameters()
		print("model: ", "MDL_C")

	elif args.backbone == "MS2CANet":
		FM = 64
		args.feature_dim = 256
		para_tune = False
		if args.dataset_name == "Houston_2013":
			para_tune = True                # para_tune 这个参数对于 Houston 的提升有两个点！！

		# model = pymodel.pyCNN(data1_bands, data2_bands, classes=class_num, \
		#                       FM=FM, para_tune=para_tune).to(args.device)
		# params = model.parameters()

		model = pymodel.pyCNN(data1_bands, data2_bands, FM=FM, para_tune=para_tune).to(args.device)
		# super_head = heads.MS2_head(args.feature_dim, class_num=class_num).to(args.device)
		# params = list(super_head.parameters())  + list(model.parameters())

	elif args.backbone == 'S2ENet':
		model = S2ENet.S2ENet(data1_bands, data2_bands, class_num, \
								patch_size=args.patch_size).to(args.device)
		params = model.parameters()

	elif args.backbone == "FusAtNet":
		model = FusAtNet.FusAtNet(data1_bands, data2_bands, class_num).to(args.device)
		params = model.parameters()

	elif args.backbone == "CrossHL":
		FM = 16
		model = CrossHL.CrossHL_Transformer(FM, data1_bands, data2_bands, class_num, \
											args.patch_size).to(args.device)
		params = model.parameters()

	elif args.backbone == "HCTNet":
		model = HCTNet(in_channels=1, num_classes=class_num).to(args.device)
		params = model.parameters()

	elif args.backbone == "SHNet":
		FM = 64
		# FM = 16
		model = SHNet.SHNet(data1_bands, data2_bands, feature=FM, \
							num_classes=class_num, factors=args.factors).to(args.device)
		params = model.parameters()

	elif args.backbone == "DSHFNet":
		model = DSHF(l1=data1_bands, l2=data2_bands, \
					num_classes=class_num, encoder_embed_dim=64).to(args.device)
		params = model.parameters()

	elif args.backbone == "MIViT":
		model = MMA.MMA(l1=data1_bands, l2=data2_bands, patch_size=args.patch_size, \
					num_patches=64, num_classes=class_num,
					encoder_embed_dim=64, decoder_embed_dim=32, en_depth=5, \
					en_heads=4, de_depth=5, de_heads=4, mlp_dim=8, dropout=0.1, \
					emb_dropout=0.1,fusion=args.fusion).to(args.device)
		params = model.parameters()
		
		# criterion = FocalLoss(loss_weight, gamma=2, alpha=None)

	elif args.backbone == "EMamba":
		model = MultimodalClassier(l1=data1_bands, l2=data2_bands,
							dim=data1_bands, num_classes=class_num).to(args.device)
		params = model.parameters()
		
	else:
		raise NotImplementedError("No models")
	print("backbone: ", args.backbone)

	return model, params

