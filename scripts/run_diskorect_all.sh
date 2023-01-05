tmux new -d \
'selems=("cross7" "bsquare" "bdiamond" "bcomplex");\
operations=("berosion" "bdilation");\
for operation in ${operations[@]}; do
	for sel in ${selems[@]}; do
		export PYTHONPATH=src
		python -m tasks.train \
			--filter_size 7\
			--epochs 10\
			--gpu 0\
			--patience 10\
			--vis_freq 125\
			--loss mse\
			--op $operation\
			--sel $sel\
			--experiment diskorect_erodila\
			smorphnetbinary\
			diskorect ""
	done
done'\
& 
tmux new -d \
'selems=("cross7" "bsquare" "bdiamond" "bcomplex");\
operations=("bopening" "bclosing");\
for operation in ${operations[@]}; do
	for sel in ${selems[@]}; do
		export PYTHONPATH=src
		python -m tasks.train \
			--filter_size 7\
			--epochs 10\
			--gpu 0\
			--patience 10\
			--vis_freq 125\
			--loss mse\
			--op $operation\
			--sel $sel\
			--experiment diskorect_opeclos\
			smorphnetbinarydouble\
			diskorect ""
	done
done'
