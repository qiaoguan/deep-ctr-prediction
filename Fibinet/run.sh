export HADOOP_HDFS_HOME=$HADOOP_HOME/../hadoop-hdfs
CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob) python train.py -ne 2