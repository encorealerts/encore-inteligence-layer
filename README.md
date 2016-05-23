Instructions for deploying to AWS ECS
=============================

Docker
------

### Generate a ECR authentication command from the AWS credentials in the computer and run it
#### Your credentials must be set on ~/.aws/credentials

aws ecr get-login --region us-east-1 | sh -

### Build the Docker image from the Dockerfile in the current directory
docker build -t meltwater/executive_alerts_intelligence .

### Add a tag "latest" to the image
docker tag meltwater/executive_alerts_intelligence:latest 421268985564.dkr.ecr.us-east-1.amazonaws.com/meltwater/executive_alerts_intelligence:latest

### Push the docker image to the AWS ECR
docker push 421268985564.dkr.ecr.us-east-1.amazonaws.com/meltwater/executive_alerts_intelligence:latest

AWS ECS
-------

* On the [AWS console](http://aws.amazon.com), access **EC2 Container Service**
* Access **Task Defitions**
* Access **executive_alerts_intelligence_task**
* Select the checkbox for the *active* task (there may be older version of this task with status *inactive*)
* In the **Actions** list, select *Update Service*
* Select the desired Cluster and Service according to the environment (test, staging or production)
* Select *Update Service*

