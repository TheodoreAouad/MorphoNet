selems=("cross3" "cross7" "diskaa2" "diskaa3" "diamondaa3" "complex")
dmodels=("lmorphnet_double" "pconvnet_double" "smorphnet_double")
smodels=("lmorphnet" "pconvnet" "smorphnet")
dop=("closing" "opening")
sop=("dilation" "erosion")

for op in ${dop[@]}; do
	for model in ${dmodels[@]}; do
		for sel in ${selems[@]}; do
			echo "Runing" $model "with" $sel
			python train.py --filter_size 7\
				--out_dir ../out\
				--epochs 1000\
				--gpu 1\
				--patience 10\
				models.$model\
				mse\
				$op\
				$sel\
				mnist ../mnist
		done
	done
done

for op in ${sop[@]}; do
	for model in ${smodels[@]}; do
		for sel in ${selems[@]}; do
			echo "Runing" $model "with" $sel
			python train.py --filter_size 7\
				--out_dir ../out\
				--epochs 1000\
				--gpu 1\
				--patience 10\
				models.$model\
				mse\
				$op\
				$sel\
				mnist ../mnist
		done
	done
done
