tmux new -d \
'selems=("cross7" "bsquare" "bdiamond" "bcomplex");\
operations=("berosion" "bdilation");\
for operation in ${operations[@]}; do
	for sel in ${selems[@]}; do
		export PYTHONPATH=src
		python -m tasks.train \
			--filter_size 7\
			--epochs 1000\
			--gpu 1\
			--patience 10\
			--vis_freq 125\
			--loss mse\
			--op $operation\
			--sel $sel\
			--percentage 40\
			--experiment baseline_3\
			smorphnet\
			mnist ./data/mnist
	done
done'\
& tmux new -d \
'selems=("cross7" "bsquare" "bdiamond" "bcomplex");\
operations=("bopening" "bclosing");\
for operation in ${operations[@]}; do
	for sel in ${selems[@]}; do
		export PYTHONPATH=src
		python -m tasks.train \
			--filter_size 7\
			--epochs 1000\
			--gpu 2\
			--patience 10\
			--vis_freq 125\
			--loss mse\
			--op $operation\
			--sel $sel\
			--percentage 40\
			--experiment baseline_3\
			smorphnetdouble\
			mnist ./data/mnist
	done
done'
