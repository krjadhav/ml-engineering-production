## Overview

AutoML helps developers with limited ML expertise train high quality image recognition models. Once you upload images to the AutoML UI, you can train a model that will be immediately available on Google Cloud for generating predictions via an easy to use REST API.

In this lab, you do the following:

- Uploading a labeled dataset to Cloud Storage and connecting it to AutoML with a CSV label file.
- Training a model with AutoML and evaluating its accuracy.
- Generating predictions on your trained model.

## Activate Cloud Shell

Cloud Shell is a virtual machine that is loaded with development tools. It offers a persistent 5GB home directory and runs on the Google Cloud. Cloud Shell provides command-line access to your Google Cloud resources.

Click **Activate Cloud Shell** icon at the top of the Google Cloud console.

When you are connected, you are already authenticated, and the project is set to your `PROJECT_ID`. The output contains a line that declares the `PROJECT_ID` for this session:

```
Your Cloud Platform project in this session is set to YOUR_PROJECT_ID
```

`gcloud` is the command-line tool for Google Cloud. It comes pre-installed on Cloud Shell and supports tab-completion.

(Optional) You can list the active account name with this command:

```
gcloud auth list
```

Click **Authorize**.

Your output should now look like this:

Output:

```
ACTIVE: *
ACCOUNT: student-01-xxxxxxxxxxxx@qwiklabs.net
To set the active account, run:
    $ gcloud config set account `ACCOUNT`
```

(Optional) You can list the project ID with this command:

```
gcloud config list project
```

Output:

```
[core]
project = <project_ID>
```

Example output:

```
[core]
project = qwiklabs-gcp-44776a13dea667a6
```

Note: For full documentation of `gcloud`, in Google Cloud, refer to the [gcloud CLI overview guide](https://cloud.google.com/sdk/gcloud).

## Task 1. Set up AutoML

AutoML provides an interface for all the steps in training an image classification model and generating predictions on it. Start by enabling the Cloud AutoML API.

From the Navigation menu, select **APIs & Services > Library**.

In the search bar type in "Cloud AutoML".

Observe the Cloud AutoML API is in the Enable state.

In a new browser, open the AutoML UI.

1. **Create storage bucket**
Now create a storage bucket by running the following:

```
gsutil mb -p $GOOGLE_CLOUD_PROJECT \
    -c standard    \
    -l us-central1 \
    gs://$GOOGLE_CLOUD_PROJECT-vcm/
```

2. In the Google Cloud console, open the Navigation menu and click on Cloud Storage to see it.

## Task 2. Upload training images to Cloud Storage

In order to train a model to classify images of clouds, you need to provide labelled training data so the model can develop an understanding of the image features associated with different types of clouds. In this example your model will learn to classify three different types of clouds: cirrus, cumulus, and cumulonimbus. To use AutoML you need to put your training images in Cloud Storage.

1. Before adding the cloud images, create an environment variable with the name of your bucket.
Run the following command in Cloud Shell:

```
export BUCKET=$GOOGLE_CLOUD_PROJECT-vcm
```

The training images are publicly available in a Cloud Storage bucket.

2. Use the gsutil command line utility for Cloud Storage to copy the training images into your bucket:

```
gsutil -m cp -r gs://spls/gsp223/images/* gs://${BUCKET}
```

3. When the images finish copying, click the Refresh button at the top of the Storage browser, then click on your bucket name. You should see 3 folders of photos for each of the 3 different cloud types to be classified.
If you click on the individual image files in each folder you can see the photos you'll be using to train your model for each type of cloud.

## Task 3. Create a dataset

Now that your training data is in Cloud Storage, you need a way for AutoML to access it. You'll create a CSV file where each row contains a URL to a training image and the associated label for that image. This CSV file has been created for you; you just need to update it with your bucket name.

1. Run the following command to copy the file to your Cloud Shell instance:

```
gsutil cp gs://spls/gsp223/data.csv .
```

2. Then update the CSV with the files in your project:

```
sed -i -e "s/placeholder/${BUCKET}/g" ./data.csv
```

3. Now upload this file to your Cloud Storage bucket:

```
gsutil cp ./data.csv gs://${BUCKET}
```

4. Once that command completes, click the Refresh button at the top of the Storage browser. Confirm that you see the `data.csv` file in your bucket.

5. Open the [Vertex AI Dataset](https://console.cloud.google.com/vertex-ai/datasets) tab.

6. At the top of the console, click **+ CREATE**.

7. Type "clouds" for the Dataset name.

8. Select Image classification (Single-label).
9. Click **CREATE**.

10. Choose **Select import files from Cloud Storage** and add the file name to the URL for the file you just uploaded - `your-bucket-name/data.csv`

An easy way to get this link is to go back to the Cloud Console, click on the `data.csv` file and then go to the URI field.

11. Click **CONTINUE**.

It will take 2 - 5 minutes for your images to import. Once the import has completed, you'll be brought to a page with all the images in your dataset.

## Task 4. Inspect images

After the import completes, you will be redirected to Browse tab to see the images you uploaded.

Try filtering by different labels in the left menu (i.e. click cumulus) to review the training images:

If any images are labeled incorrectly you can click on the image to switch the label:

## Task 5. Train your model

You're ready to start training your model! AutoML handles this for you automatically, without requiring you to write any of the model code.

1. To train your clouds model, click **TRAIN NEW MODEL**.

2. On the Training method tab, click **Continue**.

3. On the Model details tab, click **Continue**.

4. On the Training options tab, click **Continue**.

5. On the Explainability tab, click **Continue**.

6. On the Compute and pricing tab, set the node hours to 8.

7. Click **Start Training**.

Since this is a small dataset, it will only take around 25-30 minutes to complete. In the meantime, proceed to the next section to use a pre-trained model.

## Task 6. Generate predictions

There are a few ways to generate predictions. In this lab, you'll use the UI to upload images. You'll see how your model does classifying these two images (the first is a cirrus cloud, the

 second is a cumulonimbus).

1. Return to the Cloudshell terminal.

2. Download these images to your local machine.

```
gsutil cp gs://spls/gsp223/examples/* .
```

3. View the example file CLOUD1-JSON and CLOUD2-JSON to see the content.

```
{
  "instances": [{
    "content": "YOUR_IMAGE_BYTES"
  }],
  "parameters": {
    "confidenceThreshold": 0.5,
    "maxPredictions": 5
  }
}
```

Copy the Endpoint value from the Qwiklabs Panel to an environment variable.

```
ENDPOINT=$(gcloud run services describe automl-service --platform managed --region us-central1 --format 'value(status.url)')
```

5. Enter the following command to request a prediction:

```
curl -X POST -H "Content-Type: application/json" $ENDPOINT/v1 -d "@${INPUT_DATA_FILE}" | jq
```

The above call will ask AutoML for a prediction. However there is no input data specified, so the request will fail. The 400 HTTP error code indicates the expected data is not present.

Expected Output:

```
{
  "error": {
    "code": 400,
    "message": "Empty instances.",
    "status": "INVALID_ARGUMENT"
  }
}
```
