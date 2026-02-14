# 1. Run the Slime docker image
## 1.1 On private clusters (no slurm)
Pull docker image
```bash
docker pull slimerl/slime:latest
```
Run docker image
```bash
P="$(pwd)"
docker run --rm -it \
  -u "$(id -u)":"$(id -g)" \
  -v /data2/tianang:/data2/tianang \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v "$HOME":"$HOME" \
  -v "$P":"$P" \
  -e HOME="$HOME" \
  -w "$P" \
  --entrypoint bash \
  --gpus all \
  --ipc=host \
  slimerl/slime:latest
```
You can have parallel shells of this running container. To avoid creating files owned by root, start another shell with `root` identity to update the slime repo.
```bash
docker exec -it -u 0:0 <container_id> bash
```
`<container_id>` here is the hostname in `username@hostname` (e.g. `tianang@61b0797d91a1`).

In the new shell with root identity, update to the latest slime repo
```bash
cd /root/slime
git pull
pip install -e . --no-deps --break-system-packages
```
Or we can create a new folder of slime to easier manage code
```bash
cd $PWD/../..
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e . --no-deps --break-system-packages
```
## 1.2 On slurm clusters (Parcc)
1. Get a node shell without pulling container
```bash
srun -p b200-mig45 -N1 -n1 --gpus=1 --time=00:30:00 --pty bash -l
```
2. Pull the docker image
```bash
CACHE=/vast/projects/xia6/apex-gen/tianang/container_cache
mkdir -p $CACHE/{xdg_runtime,enroot_cache,enroot_data,enroot_tmp}
chmod 700 $CACHE/xdg_runtime

export XDG_RUNTIME_DIR=$CACHE/xdg_runtime
export ENROOT_CACHE_PATH=$CACHE/enroot_cache
export ENROOT_DATA_PATH=$CACHE/enroot_data
export ENROOT_TEMP_PATH=$CACHE/enroot_tmp

enroot import -o $CACHE/slime_latest.sqsh docker://slimerl/slime:latest
```
This step saves `slime_latest.sqsh` to `$CACHE/` for repeeted use.  
3. Run the docker image
```bash
CACHE=/vast/projects/xia6/apex-gen/tianang/container_cache
export XDG_RUNTIME_DIR=$CACHE/xdg_runtime
export ENROOT_CACHE_PATH=$CACHE/enroot_cache
export ENROOT_DATA_PATH=$CACHE/enroot_data
export ENROOT_TEMP_PATH=$CACHE/enroot_tmp

P="$(pwd)"

srun --partition=dgx-b200 \
     --pty \
     --nodes=1 \
     --gpus=4 \
     --cpus-per-task=64 \
     --mem=512G \
     --time=1-00:00:00 \
     --container-writable \
     --container-image=$CACHE/slime_latest.sqsh \
     --container-mounts=$P:$P,$HOME:$HOME,/vast:/vast:rw \
     --container-workdir="$P" \
     --export=ALL,HOME="$HOME" \
     bash -l
```
