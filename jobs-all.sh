#!/bin/bash

DIR_NAME=M5

#for script_name in Autoformer_M5.sh DLinear_M5.sh FedFormer_M5.sh; do
for script_name in run_FEDformer_seq12_pred12.sh run_FEDformer_seq192_pred12.sh run_FEDformer_seq192_pred144.sh run_FEDformer_seq192_pred192.sh run_FEDformer_seq192_pred24.sh run_FEDformer_seq192_pred48.sh run_FEDformer_seq192_pred96.sh run_FEDformer_seq24_pred12.sh run_FEDformer_seq24_pred24.sh run_FEDformer_seq48_pred12.sh run_FEDformer_seq48_pred24.sh run_FEDformer_seq48_pred48.sh run_FEDformer_seq96_pred12.sh run_FEDformer_seq96_pred24.sh run_FEDformer_seq96_pred48.sh run_FEDformer_seq96_pred96.sh; do
  # Format the job name to be lowercase and replace underscores and periods
  job_name=$(echo "$script_name" | tr '[:upper:]' '[:lower:]' | sed 's/[_\.]/-/g' | sed 's/\.sh//')

  # Use 'envsubst' to substitute the environment variable
  export SCRIPT_NAME="$script_name"
  export JOB_NAME="extformer-job-${job_name}"

  cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: kafka-flink
spec:
  template:
    metadata:
      labels:
        app: extformer
    spec:
      containers:
      - name: extformer
        image: ramankhurana/extformer-image:latest
        #imagePullPolicy: IfNotPresent
        resources:
          requests:
            memory: "11Gi"
            cpu: "3"
          limits:
            memory: "12Gi"
            cpu: "4"
        env:
        - name: SCRIPT_NAME
          value: "${SCRIPT_NAME}"
        - name: DIR_NAME
          value: $DIR_NAME
        volumeMounts:
        - name: dataset-volume
          mountPath: /nfs/home/khurana/dataset
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: dataset-volume
        persistentVolumeClaim:
          claimName: extformer-pvc
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: "3Gi"
      restartPolicy: Never
EOF
done

#		   run_FEDformer_seq192_pred24.sh run_FEDformer_seq192_pred48.sh run_FEDformer_seq192_pred96.sh run_FEDformer_seq24_pred12.sh run_FEDformer_seq24_pred24.sh run_FEDformer_seq336_pred12.sh run_FEDformer_seq336_pred192.sh run_FEDformer_seq336_pred24.sh run_FEDformer_seq336_pred336.sh run_FEDformer_seq336_pred48.sh run_FEDformer_seq336_pred96.sh run_FEDformer_seq48_pred12.sh run_FEDformer_seq48_pred24.sh run_FEDformer_seq48_pred48.sh run_FEDformer_seq96_pred12.sh run_FEDformer_seq96_pred24.sh run_FEDformer_seq96_pred48.sh run_FEDformer_seq96_pred96.sh; do
