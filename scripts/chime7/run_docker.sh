 docker run --gpus all -it --rm -v /home:/home --shm-size=8g \
    -p 8808:8888 -p 6066:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 --device=/dev/snd nemo-chime8