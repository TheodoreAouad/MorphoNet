selems=("cross3" "cross7" "diskaa2" "diskaa3" "diamondaa3" "complex")
dmodels=("lmorphnet_double" "pconvnet_double" "smorphnet_double")
smodels=("lmorphnet" "pconvnet" "smorphnet")
dop=("closing" "opening")
sop=("dilation" "erosion")

for op in ${dop[@]}; do
	echo $op >> "script.log"
	for model in ${dmodels[@]}; do
		echo -e "\t$model" >> "script.log"
		for sel in ${selems[@]}; do
			echo -ne "\t\t$sel" >> "script.log"
			python train.py --filter_size 7\
				--out_dir ../out\
				--epochs 1000\
				--gpu 1\
				--patience 10\
    				--vis_freq 125\
				models.$model\
				mse\
				$op\
				$sel\
				mnist ../mnist
			echo " [DONE]" >> "script.log"
		done
	done
done

for op in ${sop[@]}; do
	echo $op >> "script.log"
	for model in ${smodels[@]}; do
		echo -e "\t$model" >> "script.log"
		for sel in ${selems[@]}; do
			echo -ne "\t\t$sel" >> "script.log"
			python train.py --filter_size 7\
				--out_dir ../out\
				--epochs 1000\
				--gpu 1\
				--patience 10\
    				--vis_freq 125\
				models.$model\
				mse\
				$op\
				$sel\
				mnist ../mnist
			echo " [DONE]" >> "script.log"
		done
	done
done
