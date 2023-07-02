# TensorFlow and Vertex AI Workbench

TensorFlow is an end-to-end open-source platform for machine learning. It provides a comprehensive ecosystem of tools, libraries, and community resources that allow researchers to push the state-of-the-art in ML and developers to easily build and deploy ML-powered applications.

Vertex AI brings together AutoML and AI Platform into a unified API, client library, and user interface. With Vertex AI, both AutoML training and custom training are available options.

Vertex AI Workbench is a tool that helps users quickly build end-to-end notebook-based workflows through deep integration with data services (such as Dataproc, Dataflow, BigQuery, and Dataplex) and Vertex AI. It enables data scientists to connect to Google Cloud data services, analyze datasets, experiment with different modeling techniques, deploy trained models into production, and manage MLOps through the model lifecycle.

In this lab, we will focus on using Vertex AI Workbench to perform various tasks related to deploying a model and making predictions. The objectives of the lab include:

1. Deploying a Vertex AI Workbench instance.
2. Creating minimal training and validation data.
3. Creating the input data pipeline.
4. Creating a TensorFlow model.
5. Deploying the model to Vertex AI.
6. Deploying an Explainable AI model to Vertex AI.
7. Making predictions from the model endpoint.

## Task 1. Create minimal training and validation data

First, we need to import the required Python libraries and set environment variables. Then, we will create minimal training and validation datasets using BigQuery and export them to CSV files on Google Cloud Storage.

```python
import os, json, math, shutil
import numpy as np
import tensorflow as tf

# Environment variables used by bash cells
PROJECT = !(gcloud config get-value project)
PROJECT = PROJECT[0]
REGION = ''
BUCKET = '{}-dsongcp'.format(PROJECT)
os.environ['ENDPOINT_NAME'] = 'flights'
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['TF_VERSION'] = '2-' + tf.__version__[2:3]
```

Next, we will execute some BigQuery queries to create temporary tables containing the required data. These tables will be exported to CSV files on Google Cloud Storage.

```python
%%bigquery
CREATE OR REPLACE TABLE dsongcp.flights_train_data AS
SELECT
  IF(arr_delay < 15, 1.0, 0.0) AS ontime,
  dep_delay,
  taxi_out,
  distance,
  origin,
  dest,
  EXTRACT(hour FROM dep_time) AS dep_hour,
  IF (EXTRACT(dayofweek FROM dep_time) BETWEEN 2 AND 6, 1, 0) AS is_weekday,
  UNIQUE_CARRIER AS carrier,
  dep_airport_lat,
  dep_airport_lon,
  arr_airport_lat,
  arr_airport_lon
FROM dsongcp.flights_tzcorr f
JOIN dsongcp.trainday t
ON f.FL_DATE = t.FL_DATE
WHERE
  f.CANCELLED = False AND 
  f.DIVERTED = False AND
  is_train_day = 'True'
```

```python
%%bigquery
CREATE OR REPLACE TABLE dsongcp.flights_eval_data AS
SELECT
  IF(arr_delay < 15, 1.0, 0.0) AS ontime,
  dep_delay,
  taxi_out,
  distance,
  origin,
  dest,
  EXTRACT(hour FROM dep_time) AS dep_hour,
  IF (EXTRACT(dayofweek FROM dep_time) BETWEEN 2 AND 6, 1, 0) AS is

_weekday,
  UNIQUE_CARRIER AS carrier,
  dep_airport_lat,
  dep_airport_lon,
  arr_airport_lat,
  arr_airport_lon
FROM dsongcp.flights_tzcorr f
JOIN dsongcp.trainday t
ON f.FL_DATE = t.FL_DATE
WHERE
  f.CANCELLED = False AND 
  f.DIVERTED = False AND
  is_train_day = 'False'
```

```python
%%bigquery
CREATE OR REPLACE TABLE dsongcp.flights_all_data AS
SELECT
  IF(arr_delay < 15, 1.0, 0.0) AS ontime,
  dep_delay,
  taxi_out,
  distance,
  origin,
  dest,
  EXTRACT(hour FROM dep_time) AS dep_hour,
  IF (EXTRACT(dayofweek FROM dep_time) BETWEEN 2 AND 6, 1, 0) AS is_weekday,
  UNIQUE_CARRIER AS carrier,
  dep_airport_lat,
  dep_airport_lon,
  arr_airport_lat,
  arr_airport_lon,
  IF (is_train_day = 'True',
      IF(ABS(MOD(FARM_FINGERPRINT(CAST(f.FL_DATE AS STRING)), 100)) < 60, 'TRAIN', 'VALIDATE'),
      'TEST') AS data_split
FROM dsongcp.flights_tzcorr f
JOIN dsongcp.trainday t
ON f.FL_DATE = t.FL_DATE
WHERE
  f.CANCELLED = False AND 
  f.DIVERTED = False
```

Finally, we will export the training, validation, and full datasets to CSV files in the Google Cloud Storage bucket.

```python
%%bash
PROJECT=$(gcloud config get-value project)
for dataset in "train" "eval" "all"; do
  TABLE=dsongcp.flights_${dataset}_data
  CSV=gs://${BUCKET}/ch9/data/${dataset}.csv
  echo "Exporting ${TABLE} to ${CSV} and deleting table"
  bq --project_id=${PROJECT} extract --destination_format=CSV $TABLE $CSV
  bq --project_id=${PROJECT} rm -f $TABLE
done
```

## Task 3. Create the input data pipeline

In this task, we will set up the input data pipeline for reading and processing the training and validation datasets.

First, we will define some variables and environment settings.

```python
# For development purposes, train for a few epochs
DEVELOP_MODE = True
NUM_EXAMPLES = 5000 * 1000

# Assign your training and validation data URIs
training_data_uri = 'gs://{}/ch9/data/train*'.format(BUCKET)
validation_data_uri = 'gs://{}/ch9/data/eval*'.format(BUCKET)

# Set up Model Parameters
NBUCKETS = 5
NEMBEDS = 3
TRAIN_BATCH_SIZE = 64
DNN_HIDDEN_UNITS = '64,32'
```

Next, we will define functions for reading and processing the datasets.

```python
def features_and_labels(features):
  label = features.pop('ontime') 
  return features, label

def read_dataset(pattern, batch_size, mode=tf.estimator.ModeKeys.TRAIN, truncate=None):
  dataset = tf.data.experimental.make_csv_dataset(pattern, batch_size, num_epochs=1)
  dataset = dataset.map(features_and_labels)
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.repeat()
  
  dataset = dataset.prefetch(1)
  
  if truncate is not None:
   

 dataset = dataset.take(truncate)
  
  return dataset
```

## Task 4. Create, train, and evaluate the TensorFlow model

In this task, we will create a TensorFlow model using the input data pipeline and train it using the training dataset.

First, we will define the feature columns for the model.

```python
import tensorflow as tf

real = {
    colname: tf.feature_column.numeric_column(colname)
    for colname in ('dep_delay,taxi_out,distance,dep_hour,is_weekday,'
                    'dep_airport_lat,dep_airport_lon,'
                    'arr_airport_lat,arr_airport_lon').split(',')
}

sparse = {
    'carrier': tf.feature_column.categorical_column_with_vocabulary_list(
        'carrier',
        vocabulary_list='AS,VX,F9,UA,US,WN,HA,EV,MQ,DL,OO,B6,NK,AA'.split(',')
    ),
    'origin': tf.feature_column.categorical_column_with_hash_bucket('origin', hash_bucket_size=1000),
    'dest': tf.feature_column.categorical_column_with_hash_bucket('dest', hash_bucket_size=1000),
}
```

Then, we will create the input layers for the model.

```python
inputs = {
    colname: tf.keras.layers.Input(name=colname, shape=(), dtype='float32')
    for colname in real.keys()
}
inputs.update({
    colname: tf.keras.layers.Input(name=colname, shape=(), dtype='string')
    for colname in sparse.keys()
})
```

Next, we will perform bucketing for the real-valued columns.

```python
latbuckets = np.linspace(20.0, 50.0, NBUCKETS).tolist()  # USA
lonbuckets = np.linspace(-120.0, -70.0, NBUCKETS).tolist()  # USA

disc = {
    'd_{}'.format(key): tf.feature_column.bucketized_column(real[key], latbuckets)
    for key in ['dep_airport_lat', 'arr_airport_lat']
}

disc.update({
    'd_{}'.format(key): tf.feature_column.bucketized_column(real[key], lonbuckets)
    for key in ['dep_airport_lon', 'arr_airport_lon']
})

sparse['dep_loc'] = tf.feature_column.crossed_column(
    [disc['d_dep_airport_lat'], disc['d_dep_airport_lon']], NBUCKETS * NBUCKETS
)

sparse['arr_loc'] = tf.feature_column.crossed_column(
    [disc['d_arr_airport_lat'], disc['d_arr_airport_lon']], NBUCKETS * NBUCKETS
)

sparse['dep_arr'] = tf.feature_column.crossed_column(
    [sparse['dep_loc'], sparse['arr_loc']], NBUCKETS ** 4
)

embed = {
    'embed_{}'.format(colname): tf.feature_column.embedding_column(col, NEMBEDS)
    for colname, col in sparse.items()
}

real.update(embed)

sparse = {
    colname: tf.feature_column.indicator_column(col)
    for colname, col in sparse.items()
}
```

Now, we will create the model using a wide-and-deep architecture.

```python
def wide_and_deep_classifier(inputs, linear_feature_columns, dnn_feature_columns, dnn_hidden_units):
    deep = tf.keras.layers.DenseFeatures(dnn_feature_columns, name='deep_inputs')(inputs)
    layers = [int(x) for x in dnn_hidden_units.split(',')]
    
    for layerno, numnodes in enumerate(layers):
        deep = tf.keras.layers.Dense(numnodes, activation='relu', name='dnn_{}'.

format(layerno + 1))(deep)
    
    wide = tf.keras.layers.DenseFeatures(linear_feature_columns, name='wide_inputs')(inputs)
    both = tf.keras.layers.concatenate([deep, wide], name='both')
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(both)
    
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = wide_and_deep_classifier(
    inputs,
    linear_feature_columns=sparse.values(),
    dnn_feature_columns=real.values(),
    dnn_hidden_units=DNN_HIDDEN_UNITS
)
```

Next, we will train and evaluate the model using the training and validation datasets.

```python
train_batch_size = TRAIN_BATCH_SIZE

if DEVELOP_MODE:
    eval_batch_size = 100
    steps_per_epoch = 3
    epochs = 2
    num_eval_examples = eval_batch_size * 10
else:
    eval_batch_size = 100
    steps_per_epoch = NUM_EXAMPLES // train_batch_size
    epochs = 10
    num_eval_examples = eval_batch_size * 100

train_dataset = read_dataset(training_data_uri, train_batch_size)
eval_dataset = read_dataset(validation_data_uri, eval_batch_size, tf.estimator.ModeKeys.EVAL, num_eval_examples)

checkpoint_path = '{}/checkpoints/flights.cpt'.format(output_dir)
shutil.rmtree(checkpoint_path, ignore_errors=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    verbose=1
)

history = model.fit(
    train_dataset,
    validation_data=eval_dataset,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    callbacks=[cp_callback]
)
```

Finally, we will visualize the model loss and accuracy.

```python
import matplotlib.pyplot as plt

nrows = 1
ncols = 2

fig = plt.figure(figsize=(10, 5))

for idx, key in enumerate(['loss', 'accuracy']):
    ax = fig.add_subplot(nrows, ncols, idx + 1)
    plt.plot(history.history[key])
    plt.plot(history.history['val_{}'.format(key)])
    plt.title('model {}'.format(key))
    plt.ylabel(key)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

plt.show()
```

## Task 5. Export the trained model

In this task, we will save the trained model artifacts to the Google Cloud Storage bucket.

```python
import time

export_dir = '{}/export/flights_{}'.format(output_dir, time.strftime("%Y%m%d-%H%M%S"))
print('Exporting to {}'.format(export_dir))

tf.saved_model.save(model, export_dir)
```

### Task 5. Deploy flights model to Vertex AI

Vertex AI provides a fully managed, autoscaling, serverless environment for Machine Learning models. You get the benefits of paying for any compute resources (such as CPUs or GPUs) only when you are using them. Because the models are containerized, dependency management is taken care of. The Endpoints take care of traffic splits, allowing you to do A/B testing in a convenient way.

The benefits go beyond not having to manage infrastructure. Once you deploy the model to Vertex AI, you get a lot of neat capabilities without any additional code — explainability, drift detection, monitoring, etc.

1. Create the model endpoint flights using the following code cell and delete any existing models with the same name:

```bash
%%bash
# note TF_VERSION and ENDPOINT_NAME set in 1st cell
# TF_VERSION=2-6
# ENDPOINT_NAME=flights
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
MODEL_NAME=${ENDPOINT_NAME}-${TIMESTAMP}
EXPORT_PATH=$(gsutil ls ${OUTDIR}/export | tail -1)
echo $EXPORT_PATH
# create the model endpoint for deploying the model
if [[ $(gcloud beta ai endpoints list --region=$REGION \
        --format='value(DISPLAY_NAME)' --filter=display_name=${ENDPOINT_NAME}) ]]; then
    echo "Endpoint for $MODEL_NAME already exists"
else
    echo "Creating Endpoint for $MODEL_NAME"
    gcloud beta ai endpoints create --region=${REGION} --display-name=${ENDPOINT_NAME}
fi
ENDPOINT_ID=$(gcloud beta ai endpoints list --region=$REGION \
              --format='value(ENDPOINT_ID)' --filter=display_name=${ENDPOINT_NAME})
echo "ENDPOINT_ID=$ENDPOINT_ID"
# delete any existing models with this name
for MODEL_ID in $(gcloud beta ai models list --region=$REGION --format='value(MODEL_ID)' --filter=display_name=${MODEL_NAME}); do
    echo "Deleting existing $MODEL_NAME ... $MODEL_ID "
    gcloud ai models delete --region=$REGION $MODEL_ID
done
# create the model using the parameters docker conatiner image and artifact uri
gcloud beta ai models upload --region=$REGION --display-name=$MODEL_NAME \
     --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.${TF_VERSION}:latest \
     --artifact-uri=$EXPORT_PATH
MODEL_ID=$(gcloud beta ai models list --region=$REGION --format='value(MODEL_ID)' --filter=display_name=${MODEL_NAME})
echo "MODEL_ID=$MODEL_ID"
# deploy the model to the endpoint
gcloud beta ai endpoints deploy-model $ENDPOINT_ID \
  --region=$REGION \
  --model=$MODEL_ID \
  --display-name=$MODEL_NAME \
  --machine-type=n1-standard-2 \
  --min-replica-count=1 \
  --max-replica-count=1 \
  --traffic-split=0=100
```

2. Create a test input file example_input.json using the following code:

```bash
%%writefile example_input.json
{"instances": [
  {"dep_hour": 2, "is_weekday": 1, "dep_delay": 40, "taxi_out": 17, "distance": 41, "carrier": "AS", "dep_airport_lat": 58.42527778, "dep_airport_lon": -135.7075, "arr_airport_lat": 58.35472222, "arr_airport_lon": -134.57472222, "origin": "GST", "dest": "JNU

"},
  {"dep_hour": 22, "is_weekday": 0, "dep_delay": -7, "taxi_out": 7, "distance": 201, "carrier": "HA", "dep_airport_lat": 21.97611111, "dep_airport_lon": -159.33888889, "arr_airport_lat": 20.89861111, "arr_airport_lon": -156.43055556, "origin": "LIH", "dest": "OGG"}
]}
```

3. Make a prediction from the model endpoint. Here you have input data in a JSON file called example_input.json:

```bash
%%bash
ENDPOINT_ID=$(gcloud beta ai endpoints list --region=$REGION \
              --format='value(ENDPOINT_ID)' --filter=display_name=${ENDPOINT_NAME})
echo $ENDPOINT_ID
gcloud beta ai endpoints predict $ENDPOINT_ID --region=$REGION --json-request=example_input.json
```

Here’s how client programs can invoke the model that you deployed.

Assume that they have the input data in a JSON file called example_input.json.

4. Now, send an HTTP POST request and you will get the result back as JSON:

```bash
%%bash
PROJECT=$(gcloud config get-value project)
ENDPOINT_ID=$(gcloud beta ai endpoints list --region=$REGION \
              --format='value(ENDPOINT_ID)' --filter=display_name=${ENDPOINT_NAME})
curl -X POST \
  -H "Authorization: Bearer "$(gcloud auth application-default print-access-token) \
  -H "Content-Type: application/json; charset=utf-8" \
  -d @example_input.json \
  "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT}/locations/${REGION}/endpoints/${ENDPOINT_ID}:predict"
```

### Task 6. Model explainability

Model explainability is one of the most important problems in machine learning. It's a broad concept of analyzing and understanding the results provided by machine learning models. Explainability in machine learning means you can explain what happens in your model from input to output. It makes models transparent and solves the black box problem. Explainable AI (XAI) is the more formal way to describe this.

1. Run the following code:

```bash
%%bash
model_dir=$(gsutil ls ${OUTDIR}/export | tail -1)
echo $model_dir
saved_model_cli show --tag_set serve --signature_def serving_default --dir $model_dir
```

2. Create a JSON file explanation-metadata.json that contains the metadata describing the Model's input and output for explanation. Here, you use sampled-shapley method used for explanation:

```python
cols = ('dep_delay,taxi_out,distance,dep_hour,is_weekday,' +
        'dep_airport_lat,dep_airport_lon,' +
        'arr_airport_lat,arr_airport_lon,' +
        'carrier,origin,dest')
inputs = {x: {"inputTensorName": "{}".format(x)} 
        for x in cols.split(',')}
expl = {
    "inputs": inputs,
    "outputs": {
    "pred": {
      "outputTensorName": "pred"
    }
  }
}
print(expl)
with open('explanation-metadata.json', 'w') as ofp:
    json.dump(expl, ofp, indent=2)
```

3. View the explanation-metadata.json file using the cat command:

```bash
!cat explanation-metadata.json
```

### Task 7. Invoke the deployed model

Here’s how client programs can invoke the model you deployed. Assume that they have the input data in a JSON file called example_input.json

. Now, send an HTTP POST request and you will get the result back as JSON.

Run the following code:

```bash
%%bash
PROJECT=$(gcloud config get-value project)
ENDPOINT_NAME=flights_xai
ENDPOINT_ID=$(gcloud beta ai endpoints list --region=$REGION \
              --format='value(ENDPOINT_ID)' --filter=display_name=${ENDPOINT_NAME})
curl -X POST \
  -H "Authorization: Bearer "$(gcloud auth application-default print-access-token) \
  -H "Content-Type: application/json; charset=utf-8" \
  -d @example_input.json \
  "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT}/locations/${REGION}/endpoints/${ENDPOINT_ID}:explain"
```
