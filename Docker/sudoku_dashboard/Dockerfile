FROM rocker/shiny:3.6.3

RUN apt-get update
RUN apt-get install -y libssl-dev

RUN R -e "install.packages(c('httr', 'shinythemes', 'shinyMatrix','DT'),repos='http://cran.rstudio.com/')"

RUN mkdir /srv/shiny-server/sample-apps/sudoku_solver
COPY ./R/r_shiny_app/* /srv/shiny-server/sample-apps/sudoku_solver/
COPY ./Docker/deployment_config.yaml /srv/shiny-server/sample-apps/sudoku_solver/
RUN chmod o+xwr ./srv/shiny-server/sample-apps/sudoku_solver/deployment_config.yaml

# create explicit mount point for shared docker volume and make accesible for shiny server user 'shiny'
RUN mkdir /sudoku_volume
RUN chown shiny:shiny /sudoku_volume

# NOTE: Shiny server runs on port 3838 - that will be made accessible via the docker-compose file
