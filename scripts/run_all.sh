tmux new -d \
'selems=("cross3" "cross7" "diskaa2" "diskaa3" "diamondaa3" "complex");\
dmodels=("lmorphnet_double" "pconvnet_double" "smorphnet_double");\
smodels=("lmorphnet" "pconvnet" "smorphnet");\
for model in ${dmodels[@]}; do
	echo -e "$model" >> "closing.log"
	for sel in ${selems[@]}; do
		echo -ne "\t$sel" >> "closing.log"
		python training/train.py --filter_size 7\
			--out_dir out\
			--epochs 1000\
			--gpu 1\
			--patience 10\
			--vis_freq 125\
			models.$model\
			mse\
			closing\
			$sel\
			mnist mnist
		echo " [DONE]" >> "closing.log"
	done
done'\
& tmux new -d \
'selems=("cross3" "cross7" "diskaa2" "diskaa3" "diamondaa3" "complex");\
dmodels=("lmorphnet_double" "pconvnet_double" "smorphnet_double");\
smodels=("lmorphnet" "pconvnet" "smorphnet");\
for model in ${dmodels[@]}; do
	echo -e "$model" >> "opening.log"
	for sel in ${selems[@]}; do
		echo -ne "\t$sel" >> "opening.log"
		python training/train.py --filter_size 7\
			--out_dir out\
			--epochs 1000\
			--gpu 1\
			--patience 10\
			--vis_freq 125\
			models.$model\
			mse\
			opening\
			$sel\
			mnist mnist
		echo " [DONE]" >> "opening.log"
	done
done'\
& tmux new -d \
'selems=("cross3" "cross7" "diskaa2" "diskaa3" "diamondaa3" "complex");\
dmodels=("lmorphnet_double" "pconvnet_double" "smorphnet_double");\
smodels=("lmorphnet" "pconvnet" "smorphnet");\
for model in ${smodels[@]}; do
	echo -e "$model" >> "erosion.log"
	for sel in ${selems[@]}; do
		echo -ne "\t$sel" >> "erosion.log"
		python training/train.py --filter_size 7\
			--out_dir out\
			--epochs 1000\
			--gpu 1\
			--patience 10\
			--vis_freq 125\
			models.$model\
			mse\
			erosion\
			$sel\
			mnist mnist
		echo " [DONE]" >> "erosion.log"
	done
done'\
& tmux new -d \
'selems=("cross3" "cross7" "diskaa2" "diskaa3" "diamondaa3" "complex");\
dmodels=("lmorphnet_double" "pconvnet_double" "smorphnet_double");\
smodels=("lmorphnet" "pconvnet" "smorphnet");\
for model in ${smodels[@]}; do
	echo -e "$model" >> "dilation.log"
	for sel in ${selems[@]}; do
		echo -ne "\t$sel" >> "dilation.log"
		python training/train.py --filter_size 7\
			--out_dir out\
			--epochs 1000\
			--gpu 1\
			--patience 10\
			--vis_freq 125\
			models.$model\
			mse\
			dilation\
			$sel\
			mnist mnist
		echo " [DONE]" >> "dilation.log"
	done
done'

# To kill all running servers:
# $ tmux kill-server
