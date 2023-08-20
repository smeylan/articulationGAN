docker run -e WANDB_API_KEY=${WANDB_API_KEY} -v ~/notebooks/articulationGAN:/app -it --gpus all --rm menll-torch-agan train_mnll_artic.py --data_dir data/TIMIT_padded_smallVocab/ --log_dir log_dir/ --emadir articulatory_weights/ --batch_size 2 --kernel_len 7 --architecture eiwgan --synthesizer ArticulationGAN  --vocab 'dark greasy water year' --wandb_project mnll-articulation --wandb_group articulation_test --wandb_name noArticulationCost1 --slice_len 20480 --track_q2 1 --backprop_from_Q2 1
