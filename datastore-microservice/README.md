# Datastore Microservice

This is a microservice for storing, retrieving and deleting training data for the AI model.

---

## Get All Data Sets

> GET '/crypto/training_data'

## Get a Specific Data Set

> GET '/crypto/training_data/:data_label'

## Create a new Data Set or Replace Existing

> POST '/crypto/training_data/:data_label'

## Delete a Specific Data Set

> DELETE '/crypto/training_data/:data_label'