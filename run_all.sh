echo "ft_extr_type: $1"
echo "n_mfcc_bins: $2"
python train_ofa_net.py --task normal --ft_extr_type $1 --n_mfcc_bins $2
python train_ofa_net.py --task kernel --ft_extr_type $1 --n_mfcc_bins $2
python train_ofa_net.py --task depth --phase 1 -ft_extr_type $1 --n_mfcc_bins $2
python train_ofa_net.py --task depth --phase 2 -ft_extr_type $1 --n_mfcc_bins $2
python train_ofa_net.py --task depth --phase 3 -ft_extr_type $1 --n_mfcc_bins $2
python train_ofa_net.py --task expand --phase 1 -ft_extr_type $1 --n_mfcc_bins $2
python train_ofa_net.py --task expand --phase 2 -ft_extr_type $1 --n_mfcc_bins $2
python eval_ofa_net.py --ft_extr_type $1 --n_arch 100