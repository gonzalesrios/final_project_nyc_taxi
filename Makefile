#Project's Makefile
#Manages both API and GUI builds (Monitoring and Control)

#Image name for the docker build
API_IMAGE_NAME = api_image
MONITOR_IMAGE_NAME = monitor_image

#Container names for the docker run
API_CONTAINER_NAME = api_container
MONITOR_CONTAINER_NAME = monitor_container

#Shared Resources
NETWORK_NAME = my_network
VOLUME_NAME = my_volume

#Build Both Images (API and GUI)
build:
	docker build -t $(API_IMAGE_NAME) ./api
	docker build -t $(MONITOR_IMAGE_NAME) ./monitoring

#Run Both Containers (API and GUI)
run:
	docker network create $(NETWORK_NAME) || true
	docker volume create $(VOLUME_NAME) || true
	docker run -d --name $(API_CONTAINER_NAME) --network $(NETWORK_NAME) -p 8000:8000 $(API_IMAGE_NAME)
	docker run -d --name $(MONITOR_CONTAINER_NAME) --network $(NETWORK_NAME) -p 8501:8501 $(MONITOR_IMAGE_NAME)

#Stop Both Containers and Clean Up
clean:
	docker stop $(API_CONTAINER_NAME) 
	docker stop $(MONITOR_CONTAINER_NAME) 
	docker rm $(API_CONTAINER_NAME) 
	docker rm $(MONITOR_CONTAINER_NAME)
	docker rmi $(API_IMAGE_NAME) $(MONITOR_IMAGE_NAME)
	docker network rm $(NETWORK_NAME) 
	docker volume rm $(VOLUME_NAME)