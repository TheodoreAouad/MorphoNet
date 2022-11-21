tmux new -d \
'percentages=("5" "10" "20" "30" "40" "50");\
dmodels=("lmorphnetfour" "pconvnetfour" "smorphnetfour");\
for model in ${dmodels[@]}; do
	for percentage in ${percentages[@]}; do
		export PYTHONPATH=src
		python -m tasks.train \
			--filter_size 7\
			--epochs 1000\
			--gpu 0\
			--patience 10\
			--vis_freq 125\
			--loss mse\
			--op saltpepper\
			--percentage $percentage\
			--experiment baseline_3\
			$model\
			fashionmnist ./data/fashionmnist
	done
done'\
& tmux new -d \
'percentages=("5" "10" "20" "30" "40" "50");\
dmodels=("lmorphnetdouble" "pconvnetdouble" "smorphnetdouble");\
for model in ${dmodels[@]}; do
	for percentage in ${percentages[@]}; do
		export PYTHONPATH=src
		python -m tasks.train \
			--filter_size 7\
			--epochs 1000\
			--gpu 1\
			--patience 10\
			--vis_freq 125\
			--loss mse\
			--op pepper\
			--percentage $percentage\
			--experiment baseline_3\
			$model\
			fashionmnist ./data/fashionmnist
	done
done'\
& tmux new -d \
'percentages=("5" "10" "20" "30" "40" "50");\
dmodels=("lmorphnetdouble" "pconvnetdouble" "smorphnetdouble");\
for model in ${dmodels[@]}; do
	for percentage in ${percentages[@]}; do
		export PYTHONPATH=src
		python -m tasks.train \
			--filter_size 7\
			--epochs 1000\
			--gpu 0\
			--patience 10\
			--vis_freq 125\
			--loss mse\
			--op salt\
			--percentage $percentage\
			--experiment baseline_3\
			$model\
			fashionmnist ./data/fashionmnist
	done
done'
