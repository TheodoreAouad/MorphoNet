tmux new -d \
'selems=("cross3" "cross7" "diskaa2" "diskaa3" "diamondaa3" "complex");\
dmodels=("lmorphnetdouble" "pconvnetdouble" "smorphnetdouble");\
smodels=("lmorphnet" "pconvnet" "smorphnet");\
for model in ${dmodels[@]}; do
	for sel in ${selems[@]}; do
		export PYTHONPATH=src
		python -m tasks.train \
			--filter_size 7\
			--epochs 1000\
			--gpu 0\
			--patience 10\
			--vis_freq 125\
			--loss mse\
			--op closing\
			--sel $sel\
			--percentage 40\
			--experiment baseline_3\
			$model\
			mnist ./data/mnist
	done
done'\
& tmux new -d \
'selems=("cross3" "cross7" "diskaa2" "diskaa3" "diamondaa3" "complex");\
dmodels=("lmorphnetdouble" "pconvnetdouble" "smorphnetdouble");\
smodels=("lmorphnet" "pconvnet" "smorphnet");\
for model in ${dmodels[@]}; do
	for sel in ${selems[@]}; do
		export PYTHONPATH=src
		python -m tasks.train \
			--filter_size 7\
			--epochs 1000\
			--gpu 0\
			--patience 10\
			--vis_freq 125\
			--loss mse\
			--op opening\
			--sel $sel\
			--percentage 40\
			--experiment baseline_3\
			$model\
			mnist ./data/mnist
	done
done'\
& tmux new -d \
'selems=("cross3" "cross7" "diskaa2" "diskaa3" "diamondaa3" "complex");\
dmodels=("lmorphnetdouble" "pconvnetdouble" "smorphnetdouble");\
smodels=("lmorphnet" "pconvnet" "smorphnet");\
for model in ${smodels[@]}; do
	echo -e "$model" >> "erosion.log"
	for sel in ${selems[@]}; do
		export PYTHONPATH=src
		python -m tasks.train \
			--filter_size 7\
			--epochs 1000\
			--gpu 1\
			--patience 10\
			--vis_freq 125\
			--loss mse\
			--op erosion\
			--sel $sel\
			--percentage 40\
			--experiment baseline_3\
			$model\
			mnist ./data/mnist
	done
done'\
& tmux new -d \
'selems=("cross3" "cross7" "diskaa2" "diskaa3" "diamondaa3" "complex");\
dmodels=("lmorphnetdouble" "pconvnetdouble" "smorphnetdouble");\
smodels=("lmorphnet" "pconvnet" "smorphnet");\
for model in ${smodels[@]}; do
	for sel in ${selems[@]}; do
		export PYTHONPATH=src
		python -m tasks.train \
			--filter_size 7\
			--epochs 1000\
			--gpu 1\
			--patience 10\
			--vis_freq 125\
			--loss mse\
			--op dilation\
			--sel $sel\
			--percentage 40\
			--experiment baseline_3\
			$model\
			mnist ./data/mnist
	done
done'

# To kill all running servers:
# $ tmux kill-server
