FROM rocker/shiny:3.6.3

RUN apt-get update
RUN apt-get install -y libssl-dev

RUN R -e "install.packages(c('httr', 'shinythemes', 'shinyMatrix','DT'),repos='http://cran.rstudio.com/')"

RUN mkdir /srv/shiny-server/sample-apps/sudoku_solver
COPY ./R/r_shiny_app/* /srv/shiny-server/sample-apps/sudoku_solver/

EXPOSE 3838
