conda activate kwsenv

python train_ofa_net.py --task normal > mfcc_normal.txt
python train_ofa_net.py --task kernel > mfcc_normal2kernel.txt
python train_ofa_net.py --task depth --phase 1 > mfcc_kernel2depth1.txt
python train_ofa_net.py --task depth --phase 2 > mfcc_kernel2depth2.txt
python train_ofa_net.py --task expand --phase 1 > mfcc_depth2expand1.txt
python train_ofa_net.py --task expand --phase 2 > mfcc_depth2expand2.txt
