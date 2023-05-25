python ./launch_gcat.py -environment GCATEnv -T 20 -ST [1,5,10,20] -agent GCATAgent -FA GCAT -seed 145 -emb_dim 128 -cdm_bs 128 \
-training_epoch 50 -train_bs 128 -test_bs 1024 -learning_rate 0.001 -policy_epoch 1 \
-gamma 0.5 -n_head 1 -n_block 1 -dropout_rate 0.1 -graph_block 1 \
-morl_weights [1,1,1] -CDM IRT -data_name assist2009 -gpu_no 0 -cdm_lr 0.02 -cdm_epoch 4



