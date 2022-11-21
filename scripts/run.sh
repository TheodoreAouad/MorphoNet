export PYTHONPATH=src
python -m tasks.train \
		--filter_size 7\
		--epochs 1\
		--gpu 3\
		--patience 10\
		--vis_freq 125\
		--op salt\
		--percentage 10\
		--sel cross3\
		--loss mse\
		--experiment test2\
		smorphnetdouble\
		mnist ./data/mnist
