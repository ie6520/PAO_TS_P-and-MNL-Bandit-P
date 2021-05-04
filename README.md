# RM

download docker image 
[here](https://owncloud.yuanyl.ml/s/4vLwz6DaWAM5ei4)


load docker
```sh
 docker load --input .\my_ubuntu_1.tar
```

run docker 
```sh
docker run -itd -v "Absolute-Path-to-RM:/root/RM"  -p "2222:22" --name test1 myubuntu:1 /bin/bash -c "/etc/rc.d/rc.local;service ssh restart;/bin/bash"
```

ssh connect

## Paper
[overleaf](https://www.overleaf.com/9599665731zxtrdftccmnc)
