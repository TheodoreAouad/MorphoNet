tmux new -d \
'
for i in {1..5}; do
    python training/train.py --filter_size 7\
        --out_dir out\
        --epochs 1000\
        --gpu 0\
        --patience 10\
        --vis_freq 125\
        models.smorphnet\
        mse\
        erosion\
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
        models.smorphnet\
        mse\
        erosion\
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
        models.smorphnet\
        mse\
        erosion\
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
        models.smorphnet\
        mse\
        erosion\
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
        models.smorphnet\
        mse\
        erosion\
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
        models.smorphnet\
        mse\
        erosion\
        --sel complex\
        mnist mnist
done'\
