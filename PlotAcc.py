import pickle
from matplotlib import pyplot as plt
import os
import argparse



parser = argparse.ArgumentParser(description='Training Capsules using Inverted Dot-Product Attention Routing')
parser.add_argument('--model', default='', type=str, help='Absolute Path of model where the checkpoint is saved')

args = parser.parse_args()

# -

# save_dir_name = 'model_' + str(args.model)+ '_dataset_' + str(args.dataset) + '_num_routing_' + str(args.num_routing) + '_backbone_' + args.backbone 
save_dir_name= args.model
store_dir = save_dir_name 
store_file = os.path.join(store_dir, 'debug.dct')
with open(store_file, "rb") as fp:
	results = pickle.load(fp)

print(len(results['train_acc']), len(results['test_acc']))
# -
# Plotting training and test accuracies
print("Max Train accuracy: ", max(results['train_acc']))
print("Max Test accuracy: ", max(results['test_acc']))
plt.plot(results['train_acc'], label='Training Accuracy')
plt.plot(results['test_acc'], label='Validation Accuracy')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold', rotation=90)
plt.xlabel("Epoch")
plt.tight_layout(pad=0)
plt.legend()
plt.title("Best Accuracy "+str(max(results['test_acc'])) +"%")
plot_save_path =os.path.join(store_dir, 'AccuracyPlot.png')
plt.savefig(plot_save_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.show()