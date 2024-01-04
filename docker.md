## docker running status 
docker ps


## create docker image
docker build  -t extformer-image ExtFormer

## create docker image without cache
docker build   --no-cache -t extformer-image ExtFormer

## save the image as tar file 
docker save extformer-image:latest > extformer-image.tar


## load image from tar file
docker load < extformer-image.tar
