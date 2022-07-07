echo "ft_extr_type: $1"
echo "params_id: $2"
python train_ofa_net.py --task kernel --ft_extr_type $1 --params_id $2
python train_ofa_net.py --task depth --phase 1 --ft_extr_type $1 --params_id $2
python train_ofa_net.py --task depth --phase 2 --ft_extr_type $1 --params_id $2
python train_ofa_net.py --task depth --phase 3 --ft_extr_type $1 --params_id $2
python train_ofa_net.py --task expand --phase 1 --ft_extr_type $1 --params_id $2
python train_ofa_net.py --task expand --phase 2 --ft_extr_type $1 --params_id $2
python eval_ofa_net.py --ft_extr_type $1 --params_id $2 --n_arch 1000 --use_json --measure_latency