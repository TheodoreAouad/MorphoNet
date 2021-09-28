#python train.py --filter_size 7\
#     	         --out_dir ../out\
#	         --epochs 10\
#	         --vis_freq 512\
#	         --batch_size 32\
#	         --patience 50\
#	         models.smorphnet_double\
#	         mse\
#	         closing\
#	         complex\
#	         mnist ../mnist

#python train.py --filter_size 7\
#		--out_dir ../test\
#		--epochs 1000\
#		--gpu 1\
#		--patience 10\
#		models.smorphnet_double\
#		mse\
#		closing\
#		--sel complex\
#		mnist ../mnist

#python train.py --filter_size 7\
#		--out_dir ../test\
#		--epochs 1000\
#		--gpu 1\
#		--patience 10\
#		models.pconvnet_double\
#		mse\
#		closing\
#		--sel complex\
#		mnist ../mnist

#tmux new -d
#'
#python train.py --filter_size 7\
#		--out_dir ../out_binary\
#		--epochs 1000\
#		--gpu 1\
#		--patience 10\
#		--vis_freq 125\
#		models.smorphnet\
#		mse\
#		berosion\
#		--sel bsquare\
#		mnist ../mnist
#'

python train.py --filter_size 7\
    --out_dir test_tanh\
    --epochs 1000\
    --gpu 0\
    --patience 10\
    --vis_freq 125\
    models.smorphnet_double\
    mse\
    closing\
    --sel doubledisk7_1\
    mnist ../mnist
