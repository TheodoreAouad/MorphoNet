tmux new -d \
'dmodels=("lmorphnet_double" "pconvnet_double" "smorphnet_double");\
for i in {1..5}; do
    for model in ${dmodels[@]}; do
        python training/train.py --filter_size 7\
            --out_dir out\
            --epochs 1000\
            --gpu 1\
            --patience 10\
            --vis_freq 125\
            models.$model\
            mse\
            closing\
            cross3\
            mnist mnist
    done
done'
& tmux new -d \
'dmodels=("lmorphnet_double" "pconvnet_double" "smorphnet_double");\
for i in {1..5}; do
    for model in ${dmodels[@]}; do
        python training/train.py --filter_size 7\
            --out_dir out\
            --epochs 1000\
            --gpu 1\
            --patience 10\
            --vis_freq 125\
            models.$model\
            mse\
            closing\
            cross7\
            mnist mnist
    done
done'\
& tmux new -d \
'dmodels=("lmorphnet_double" "pconvnet_double" "smorphnet_double");\
for i in {1..5}; do
    for model in ${dmodels[@]}; do
        python training/train.py --filter_size 7\
            --out_dir out\
            --epochs 1000\
            --gpu 1\
            --patience 10\
            --vis_freq 125\
            models.$model\
            mse\
            closing\
            diskaa2\
            mnist mnist
    done
done'\
& tmux new -d \
'dmodels=("lmorphnet_double" "pconvnet_double" "smorphnet_double");\
for i in {1..5}; do
    for model in ${dmodels[@]}; do
        python training/train.py --filter_size 7\
            --out_dir out\
            --epochs 1000\
            --gpu 2\
            --patience 10\
            --vis_freq 125\
            models.$model\
            mse\
            closing\
            diskaa3\
            mnist mnist
    done
done'\
& tmux new -d \
'dmodels=("lmorphnet_double" "pconvnet_double" "smorphnet_double");\
for i in {1..5}; do
    for model in ${dmodels[@]}; do
        python training/train.py --filter_size 7\
            --out_dir out\
            --epochs 1000\
            --gpu 2\
            --patience 10\
            --vis_freq 125\
            models.$model\
            mse\
            closing\
            diamondaa3\
            mnist mnist
    done
done'\
& tmux new -d \
'dmodels=("lmorphnet_double" "pconvnet_double" "smorphnet_double");\
for i in {1..5}; do
    for model in ${dmodels[@]}; do
        python training/train.py --filter_size 7\
            --out_dir out\
            --epochs 1000\
            --gpu 2\
            --patience 10\
            --vis_freq 125\
            models.$model\
            mse\
            closing\
            complex\
            mnist mnist
    done
done'\
