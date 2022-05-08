max=5
for i in `seq 1 $max`
do
    python dqn/dqn/box2d.py --log_dir logs/vanilla-dqn-exp1 --target-update-period 50 --buffer-capacity 10000

    python dqn/dqn/box2d.py --log_dir logs/vanilla-dqn-exp2 --target-update-period 300 --buffer-capacity 50000

    python dqn/dqn/box2d.py --log_dir logs/vanilla-dqn-exp3 --target-update-period 250 --buffer-capacity 40000 --epsilon-decay 0.99

    python dqn/rainbow/box2d.py --n-iterations 5000 --no-double --no-dueling --no-noisy --no-prioritized --n-steps 1 --no-dist

    python dqn/rainbow/box2d.py --log_dir logs/prioritized --no-dist --no-dueling --n-step 1 --no-double --no-noisy

    python dqn/rainbow/box2d.py --log_dir logs/distributional --no-prioritized --no-dueling --n-step 1 --no-double --no-noisy

    python dqn/rainbow/box2d.py --log_dir logs/nsteps --no-prioritized --no-dist --no-dueling --n-step 5 --no-double --no-noisy

    python dqn/rainbow/box2d.py --log_dir logs/double --no-prioritized --no-dist --no-dueling --n-step 1 --no-noisy

    python dqn/rainbow/box2d.py --log_dir logs/dueling --no-prioritized --no-dist --n-step 5 --no-double --no-noisy

    python dqn/rainbow/box2d.py --log_dir logs/noisy --no-prioritized --no-dist --no-dueling --n-step 1 --no-double
done

    