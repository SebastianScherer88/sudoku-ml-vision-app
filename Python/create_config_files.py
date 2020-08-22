# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 15:50:22 2020

@author: bettmensch
"""

import yaml

# [1] build micro service config file
python_api = {'service':'model_api',
              'container_port':'8000',
              'host_port':'8000',
              'host':'0.0.0.0'} # hardcoded in sudoku_model_api docker file

dashboard = {'service':'dashboard',
             'container_port':'3838',
              'host_port':'3838'} # hardcoded in security configuration of the hosting EC2 instance/image

storage = {'s3_bucket':'bettmensch88-aws-dev-bucket',
           's3_directory':'sudoku_solver_application'}

deployment_config = {'python_api': python_api,
                     'dashboard':dashboard,
                     'storage':storage}

# export micro service configs to both micro services' subdirs
with open('../deployment_config.yaml','w') as config_file:
    yaml.dump(deployment_config,config_file)
    
# [2] build docker-compose file
docker_compose_yaml = {}

model_api_command = '["uvicorn", "sudoku_solver_api:app", "--reload", "--host", "' + str(python_api['host']) + '", "--port", "'+ str(python_api['container_port']) + '"]'
dashboard_command = '["/usr/bin/shiny-server.sh"]'

docker_compose_yaml['version'] = '3.8'
docker_compose_yaml['services'] = {
        python_api['service']:{
                'image':'bettmensch88/sudoku_model_api:latest',
                'volumes': [{'sudoku_volume':'/sudoku_volume'}],
                'ports': [''+python_api['host_port']+':'+python_api['container_port']+''],
                'command':model_api_command},
        dashboard['service']:{
                'image':'bettmensch88/sudoku_dashboard:latest',
                'volumes': [{'sudoku_volume':'/sudoku_volume'}],
                'ports': [dashboard['host_port']+':'+dashboard['container_port']],
                'command':dashboard_command}
        }
docker_compose_yaml['volumes'] = {'sudoku_volume':''}

with open('../Docker/docker-compose.yaml','w') as docker_compose_file:
    yaml.dump(docker_compose_yaml,docker_compose_file,default_flow_style=False)