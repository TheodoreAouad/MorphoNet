export PYTHONPATH=src
python -m tasks.train \
		--filter_size 7\
		--epochs 10\
		--gpu 0\
		--patience 10\
		--vis_freq 125\
		--op bdilation\
		--percentage 10\
		--sel cross7\
		--loss mse\
		--experiment binary_mnist\
		smorphnet\
		mnist ./data/mnist


tasks.train \
		"--filter_size","7","--epochs","10","--gpu","0","--patience","10","--vis_freq","125","--op","salt","--percentage","10","--sel","cross3","--loss","mse","--experiment","test2","smorphnetdoublenist","./data/mnist"