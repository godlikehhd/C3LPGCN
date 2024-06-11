# C3LPGCN

Code and datasets of our paper "[C3LPGCN:Integrating Contrastive Learning and Cooperative Learning with Prompt into Graph Convolutional Network for Aspect-based Sentiment Analysis]" accepted by Findings of NAACL 2024.


## Preparation

1. Download the best model [best_parser.pt](LAL-Parser/best_model/readme.md) of [LAL-Parser](https://github.com/KhalilMrini/LAL-Parser).

## Training

To train the model, run:

'python train.py --model_name clpt --dataset laptop --seed 1000 --bert_lr 3e-5 --num_epoch 10 --hidden_dim 768 --max_length 125 --cuda 0 --batch_size 32 --rnn_layers 3 --num_layers 1 --input_dropout 0.3 --gcn_dropout 0.0 --bert_dropout 0.2 --rnn_dropout 0.0 --l1 0.6 --l3 0.1 --l4 0.1 --l5 0.1 --l6 0.1 --temp 0.05'

'python train.py --model_name clpt --dataset restaurant --seed 1000 --bert_lr 3e-5 --num_epoch 10 --hidden_dim 768 --max_length 125 --cuda 0 --batch_size 32 --rnn_layers 2 --num_layers 1 --input_dropout 0.3 --gcn_dropout 0.0 --bert_dropout 0.0 --rnn_dropout 0.1 --l1 0.2 --l3 0.1 --l4 0.1 --l5 0.1 --l6 0.3 --temp 0.05'

'python train.py --model_name clpt --dataset twitter --seed 1000 --bert_lr 2e-5 --num_epoch 10 --hidden_dim 768 --max_length 125 --cuda 0 --batch_size 32 --rnn_layers 3 --num_layers 1 --input_dropout 0.5 --gcn_dropout 0.1 --bert_dropout 0.3 --rnn_dropout 0.1'




## Credits

The code and datasets in this repository are based on [DualGCN-ABSA](https://github.com/CCChenhao997/DualGCN-ABSA).



