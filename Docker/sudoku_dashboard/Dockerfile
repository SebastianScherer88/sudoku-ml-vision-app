FROM rocker/shiny:3.6.3

RUN apt-get update
RUN apt-get install -y libssl-dev

RUN R -e "install.packages('devtools')"
RUN R -e "devtools::install_version('shiny', version = '1.3.2', repos = 'http://cran.us.r-project.org',upgrade = 'never')"
RUN R -e "devtools::install_version('shinythemes', version = '1.1.2', repos = 'http://cran.us.r-project.org',upgrade = 'never')"
RUN R -e "devtools::install_version('shinyMatrix', version = '0.3.0', repos = 'http://cran.us.r-project.org',upgrade = 'never')"
RUN R -e "devtools::install_version('DT', version = '0.13', repos = 'http://cran.us.r-project.org',upgrade = 'never')"

RUN mkdir /srv/shiny-server/sample-apps/sudoku_solver
COPY ./R/r_shiny_app/* /srv/shiny-server/sample-apps/sudoku_solver/
COPY ./Docker/deployment_config.yaml /srv/shiny-server/sample-apps/sudoku_solver/
RUN chown shiny:shiny ./srv/shiny-server/sample-apps/sudoku_solver/deployment_config.yaml

# create explicit mount point for shared docker volume and make accesible for shiny server user 'shiny'
RUN mkdir /sudoku_volume
RUN chown shiny:shiny /sudoku_volume

# NOTE: Shiny server runs on port 3838 - that will be made accessible via the docker-compose file
