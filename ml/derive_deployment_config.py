# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 21:30:56 2020
This script derives a deployment config file from the docker compose file to ensure the code reflects the micro service specs
of the docker network when making cross-container service calls.
@author: bettmensch
"""

import yaml

with open('../Docker/docker-compose.yaml','r') as docker_compose_file:
    docker_compose_config = yaml.load(docker_compose_file,Loader=yaml.FullLoader)
    
deployment_config = {}

model_api_specs = docker_compose_config['services']['model_api']
dashboard_specs = docker_compose_config['services']['dashboard']

model_api_config = {'service_name':'model_api',
                    'host_port':model_api_specs['ports'][0].split(':')[0],
                    'container_port':model_api_specs['ports'][0].split(':')[1],
                    'shared_drive':model_api_specs['volumes'][0].split(':')[1]}

dashboard_config = {'service_name':'dashboard',
                    'host_port':dashboard_specs['ports'][0].split(':')[0],
                    'container_port':dashboard_specs['ports'][0].split(':')[1],
                    'shared_drive':dashboard_specs['volumes'][0].split(':')[1]}

deployment_config = {'api':model_api_config,
                     'dashboard':dashboard_config}

with open('../Docker/deployment_config.yaml','w') as deployment_file:
    yaml.dump(deployment_config,deployment_file)