conda activate kwsenv

python train_ofa_net.py --task normal --ft_extr_type mfcc 2>&1 | tee mfcc_normal.txt
python train_ofa_net.py --task kernel --ft_extr_type mfcc 2>&1 | tee mfcc_normal2kernel.txt
python train_ofa_net.py --task depth --phase 1 --ft_extr_type mfcc 2>&1 | tee mfcc_kernel2depth1.txt
python train_ofa_net.py --task depth --phase 2 --ft_extr_type mfcc 2>&1 | tee mfcc_kernel2depth2.txt
python train_ofa_net.py --task expand --phase 1 --ft_extr_type mfcc 2>&1 | tee mfcc_depth2expand1.txt
python train_ofa_net.py --task expand --phase 2 --ft_extr_type mfcc 2>&1 | tee mfcc_depth2expand2.txt
