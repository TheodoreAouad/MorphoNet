tmux new -d \
'
for i in {1..5}; do
    python training/train.py --filter_size 7\
        --out_dir out\
        --epochs 1000\
        --gpu 1\
        --patience 10\
        --vis_freq 125\
        models.lmorphnet_double\
        mse\
        opening\
        cross3\
        mnist mnist
done'\
& tmux new -d \
'
for i in {1..5}; do
    python training/train.py --filter_size 7\
        --out_dir out\
        --epochs 1000\
        --gpu 0\
        --patience 10\
        --vis_freq 125\
        models.lmorphnet_double\
        mse\
        opening\
        cross7\
        mnist mnist
done'\
& tmux new -d \
'
for i in {1..5}; do
    python training/train.py --filter_size 7\
        --out_dir out\
        --epochs 1000\
        --gpu 0\
        --patience 10\
        --vis_freq 125\
        models.lmorphnet_double\
        mse\
        opening\
        diskaa2\
        mnist mnist
done'\
& tmux new -d \
'
for i in {1..5}; do
    python training/train.py --filter_size 7\
        --out_dir out\
        --epochs 1000\
        --gpu 0\
        --patience 10\
        --vis_freq 125\
        models.lmorphnet_double\
        mse\
        opening\
        diskaa3\
        mnist mnist
done'\
& tmux new -d \
'
for i in {1..5}; do
    python training/train.py --filter_size 7\
        --out_dir out\
        --epochs 1000\
        --gpu 0\
        --patience 10\
        --vis_freq 125\
        models.lmorphnet_double\
        mse\
        opening\
        diamondaa3\
        mnist mnist
done'\
& tmux new -d \
'
for i in {1..5}; do
    python training/train.py --filter_size 7\
        --out_dir out\
        --epochs 1000\
        --gpu 0\
        --patience 10\
        --vis_freq 125\
        models.lmorphnet_double\
        mse\
        opening\
        complex\
        mnist mnist
done'\
