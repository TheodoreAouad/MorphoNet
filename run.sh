export PYTHONPATH=src
python -m tasks.train \
		--filter_size 7\
		--epochs 1000\
		--gpu 3\
		--patience 10\
		--vis_freq 125\
        --op opening\
		--sel complex\
		--loss mse\
		--experiment baseline_2\
		lmorphnetdouble\
		mnist ./data/mnist
