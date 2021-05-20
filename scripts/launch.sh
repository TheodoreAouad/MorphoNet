tmux new -d \
'python train.py --filter_size 13\
		--out_dir ../out\
		--epochs 1000\
		--gpu 0\
		--patience 10\
		--vis_freq 125\
		models.lmorphnet_double\
		mse\
		closing\
		diskaa3\
		mnist ../mnist'\
& tmux new -d \
'python train.py --filter_size 13\
		--out_dir ../out\
		--epochs 1000\
		--gpu 0\
		--patience 10\
		--vis_freq 125\
		models.lmorphnet_double\
		mse\
		opening\
		diskaa3\
		mnist ../mnist'\
& tmux new -d \
'python train.py --filter_size 13\
		--out_dir ../out\
		--epochs 1000\
		--gpu 0\
		--patience 10\
		--vis_freq 125\
		models.smorphnet_double\
		mse\
		opening\
		diskaa3\
		mnist ../mnist'\
& tmux new -d \
'python train.py --filter_size 13\
		--out_dir ../out\
		--epochs 1000\
		--gpu 0\
		--patience 10\
		--vis_freq 125\
		models.smorphnet_double\
		mse\
		closing\
		diskaa3\
		mnist ../mnist'
