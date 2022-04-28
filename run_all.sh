conda activate /home/sem22f41/FeatureExtractionKWS/env

model_list = 'dscnn dscnn_maxpool dscnn_avgpool dscnn_subconv wav2vec'
ft_extr_list = 'mfcc raw augmented'

python main.py --model dscnn --ft_extr mfcc > dscnn_mfcc.txt
python main.py --model dscnn --ft_extr augmented > dscnn_augmented.txt
python main.py --model dscnn_maxpool --ft_extr augmented > dscnn_maxpool_augmented.txt
python main.py --model dscnn_avgpool --ft_extr augmented > dscnn_avgpool_augmented.txt
python main.py --model wav2vec --ft_extr raw > wav2vec_raw.txt
python main.py --model wav2vec --ft_extr augmented > wav2vec_augmented.txt