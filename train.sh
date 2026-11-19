CUDA_VISIBLE_DEVICES=0 nohup ./tools/dist_train.sh ./my_config_synapse_new/unet_all.py 1 >> exp1.log &
CUDA_VISIBLE_DEVICES=1 nohup ./tools/dist_train.sh ./my_config_synapse_new/unet_fd_kan.py 1 >> exp2.log &
CUDA_VISIBLE_DEVICES=2 nohup ./tools/dist_train.sh ./my_config_synapse_new/unet_gls.py 1 >> exp3.log &
CUDA_VISIBLE_DEVICES=3 nohup ./tools/dist_train.sh ./my_config_synapse_new/unet_mgd.py 1 >> exp4.log &
CUDA_VISIBLE_DEVICES=0 nohup ./tools/dist_train.sh ./my_config_synapse_new/unet.py 1 >> exp5.log &

CUDA_VISIBLE_DEVICES=3 nohup ./tools/dist_train.sh ./my_config_cvc/emcad.py 1 >> exp4.log &
CUDA_VISIBLE_DEVICES=0 nohup ./tools/dist_train.sh ./my_config_cvc/mkunet.py 1 >> exp5.log &