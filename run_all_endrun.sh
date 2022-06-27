python train_ofa_net.py --task depth --phase 1 --ft_extr_type mfcc 2>&1 | tee term_logs/mfcc_kernel2depth1.txt
python train_ofa_net.py --task depth --phase 2 --ft_extr_type mfcc 2>&1 | tee term_logs/mfcc_kernel2depth2.txt
python train_ofa_net.py --task depth --phase 3 --ft_extr_type mfcc 2>&1 | tee term_logs/mfcc_kernel2depth3.txt
python train_ofa_net.py --task expand --phase 1 --ft_extr_type mfcc 2>&1 | tee term_logs/mfcc_depth2width1.txt
python train_ofa_net.py --task expand --phase 2 --ft_extr_type mfcc 2>&1 | tee term_logs/mfcc_depth2width2.txt