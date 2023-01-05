export PYTHONPATH=src
python -m tasks.train \
		--filter_size 7\
		--epochs 5\
		--gpu 0\
		--patience 10\
		--vis_freq 125\
		--loss mse\
		--op bopeningarray\
		--sel cross7\
		--experiment diskorect_test\
		smorphnetbinarydouble\
		diskorect ""
