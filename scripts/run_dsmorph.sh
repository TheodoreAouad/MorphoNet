tmux new -d \
'
for i in {1..5}; do
    python training/train.py --filter_size 7\
        --out_dir out\
        --epochs 1000\
        --gpu 0\
        --patience 10\
        --vis_freq 125\
        models.smorphnet_double\
        mse\
        closing\
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
        models.smorphnet_double\
        mse\
        closing\
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
        models.smorphnet_double\
        mse\
        closing\
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
        models.smorphnet_double\
        mse\
        closing\
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
        models.smorphnet_double\
        mse\
        closing\
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
        models.smorphnet_double\
        mse\
        closing\
        complex\
        mnist mnist
done'\
