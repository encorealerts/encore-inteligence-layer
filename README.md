Instructions for deploying to AWS ECS
=============================

Docker
------

Running locally with docker:

### Build the Docker image from the Dockerfile in the current directory
docker build -t meltwater/executive_alerts_intelligence .

### [Option] Run locally to test it
docker run --rm --name executive_alerts_intelligence_container -p 5001:5001 \
		   -e "AWS_ACCESS_KEY_ID=00000000000000000" \
		   -e "AWS_SECRET_ACCESS_KEY=00000000000000000000" \
		   -e "LUIGI_S3_BUCKET=encorealert-luigi-test" \
		   -it meltwater/executive_alerts_intelligence

### TravisCI

The project now is integrated with Travis, which automatically builds and pushes an image to ECR on each push to github.
Once the image is successfully created by Travis, the following steps must be executed to run on ECS.

AWS ECS
-------

* On the [AWS console](http://aws.amazon.com), access **EC2 Container Service**
* Access **Task Defitions**
* Access **executive_alerts_intelligence_task**
* Select the checkbox for the *active* task (there may be older version of this task with status *inactive*)
* In the **Actions** list, select *Update Service*
* Select the desired Cluster and Service according to the environment (test, staging or production)
* Select *Update Service*

