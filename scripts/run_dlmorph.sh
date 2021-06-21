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
        --sel cross3\
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
        --sel cross7\
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
        --sel diskaa2\
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
        --sel diskaa3\
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
        --sel diamondaa3\
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
        --sel complex\
        mnist mnist
done'\
