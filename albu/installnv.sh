# This is just to install nvidia-docker2


docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
 sudo apt-get purge -y nvidia-docker

 # Add the package repositories
 curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
	   sudo apt-key add -
 distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
 curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
	   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
 sudo apt-get update

 # Install nvidia-docker2 and reload the Docker daemon configuration
 sudo apt-get install -y nvidia-docker2
 sudo pkill -SIGHUP dockerd

 # Test nvidia-smi with the latest official CUDA image
 docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
