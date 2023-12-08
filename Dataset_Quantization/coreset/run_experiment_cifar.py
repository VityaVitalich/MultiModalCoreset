from validate_cifar import run_experiment
import os

ratios_acc = []
if __name__ == '__main__':
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    for mask_ratio in ratios:
        saved_path = f'./results/pixel_quantized_cifar_{mask_ratio}'
        if not os.path.exists(saved_path):
            os.makedirs(saved_path, exist_ok=True)

            os.system(f'python -u quantize_pixel.py     --data CIFAR10 --data_path /home/data/dq/cifar-10-batches-py/  \
            --output_dir ./results/pixel_quantized_cifar_{mask_ratio} --model mae_vit_large_patch16   \
                --resume ./pretrained/mae_visualize_vit_large_ganloss.pth --batch_size 512     --mask_ratio {mask_ratio} --cam_mask')

        
        select_indices_path = './results/sample_quantized_cifar_010/select_indices_CIFAR10_0.6.npy'
        best_acc = run_experiment(lr=0.1,
                        bs=128,
                        data_dir=saved_path,
                        selected_indices=[select_indices_path], 
                        result_path='')
        ratios_acc.append(best_acc)
    
    print(ratios_acc)
