import pickle
from matplotlib import pyplot as plt
import os
import argparse



parser = argparse.ArgumentParser(description='Training Capsules using Inverted Dot-Product Attention Routing')

parser.add_argument('--num_routing', default=2, type=int, help='number of routing. Recommended: 0,1,2,3.')
parser.add_argument('--dataset', default='AffNIST', type=str, help='dataset. CIFAR10 or CIFAR100.')
parser.add_argument('--backbone', default='resnet', type=str, help='type of backbone. simple or resnet')
parser.add_argument('--model', default='sinkhorn', type=str, help='default or sinkhorn')
args = parser.parse_args()

# -

# save_dir_name = 'model_' + str(args.model)+ '_dataset_' + str(args.dataset) + '_num_routing_' + str(args.num_routing) + '_backbone_' + args.backbone 
save_dir_name="model_bilinear_dataset_Expanded_AffNISTv2_batch_128_acc_1.0_epochs_400_optimizer_SGD_scheduler_StepLR_steps_5_gamma_0.1_num_routing_2_backbone_resnet_config_64_sequential_routing_False_augment_0.15_0.15_0"
store_dir = os.path.join('results/AffNIST/', save_dir_name) 
store_file = os.path.join(store_dir, 'debug_replica2.dct')
with open(store_file, "rb") as fp:
	results = pickle.load(fp)

print(len(results['train_acc']), len(results['test_acc']))
# -
# Plotting training and test accuracies
print("Max Test accuracy: ", max(results['test_acc']))
plt.plot(results['train_acc'], label='Training Accuracy')
plt.plot(results['test_acc'], label='Validation Accuracy')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold', rotation=90)
plt.xlabel("Epoch")
plt.tight_layout(pad=0)
plt.legend()
plt.title("Best Accuracy on AffNIST "+str(max(results['test_acc'])) +"%")
plot_save_path =os.path.join(store_dir, 'AccuracyPlot.png')
plt.savefig(plot_save_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.show()