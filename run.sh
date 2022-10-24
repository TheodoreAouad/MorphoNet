export PYTHONPATH=src
python -m tasks.train \
		--filter_size 7\
		--epochs 1000\
		--gpu 0\
		--patience 10\
		--vis_freq 125\
        --op erosion\
		--sel complex\
		--loss mse\
		--experiment testdefou\
		smorphnet\
		mnist ./data/mnist
