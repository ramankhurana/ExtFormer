# ExtFormer 
ExtFormer is an open-source library focusing on time series forecasting using transformer variants. The existing transformer models are extended to use the static dataset which is generally useful to make accurate predictions.


The existing transformer models does not take care of sttaic data intrinsicly in the model architenture. The goal here is to extend the models architecture which emable them to use the static data and improve the predictions.

The codebase is motivated from Time Series Analysis (developer of TimesNet).

- [Disclaimer] This is a work in progress.OBBBB




## Before creating docker container prepare all the shell scripts in the script/long_forecasting/M5(Divvy)

## To do for next version
- add boolean for the static variables
- add static data in the image OR mounted on the dataset pvc
- change the static module such that it takes seq and pred length instead of 96


# create docker container
docker build  -t extformer-image ExtFormer

# create docker container without cache
docker build  --no-cache -t extformer-image ExtFormer

# login to docker
docker login

# tag the image just created
docker tag extformer-image:latest ramankhurana/extformer-image:latest

# push the image to dockerhub
docker push ramankhurana/extformer-image:latest


# run the kubernetes job
kubectl apply -f job.yaml

# run all the jobs; one job for each shell script
source jobs-all.sh


#
# delete the batch jobs before submitting them again

