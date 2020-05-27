library(shinyMatrix)
library(shinythemes)
library(shiny)

# Global variables can go here
n <- 200

ui <- fluidPage(
  theme = shinytheme('darkly'),
  column(10,
    fluidRow(
      column(1,offset=0,
             div(style = "padding: 0px 0px; margin-top:0em",
                 fluidRow(
                   matrixInput('m1', value = matrix(data = '',ncol=3,nrow=3))
                 )
             ),
             div(style = "padding: 0px 0px; margin-top:-15%",
                 fluidRow(
                   matrixInput('m2', value = matrix(data = '',ncol=3,nrow=3))
                 )
             ),
             div(style = "padding: 0px 0px; margin-top:-15%",
                 fluidRow(
                   matrixInput('m3', value = matrix(data = '',ncol=3,nrow=3))
                 )
             )
      ),
      column(1,offset=0,
             div(style = "padding: 0px 0px; margin-top:0em",
                 fluidRow(
                   matrixInput('m4', value = matrix(data = '',ncol=3,nrow=3))
                 )
             ),
             div(style = "padding: 0px 0px; margin-top:-15%",
                 fluidRow(
                   matrixInput('m5', value = matrix(data = '',ncol=3,nrow=3))
                 )
             ),
             div(style = "padding: 0px 0px; margin-top:-15%",
                 fluidRow(
                   matrixInput('m6', value = matrix(data = '',ncol=3,nrow=3))
                 )
             )
      ),
      column(1,offset=0,
             div(style = "padding: 0px 0px; margin-top:0em",
                 fluidRow(
                   matrixInput('m7', value = matrix(data = '',ncol=3,nrow=3))
                 )
             ),
             div(style = "padding: 0px 0px; margin-top:-15%",
                 fluidRow(
                   matrixInput('m8', value = matrix(data = '',ncol=3,nrow=3))
                 )
             ),
             div(style = "padding: 0px 0px; margin-top:-15%",
                 fluidRow(
                   matrixInput('m9', value = matrix(data = '',ncol=3,nrow=3))
                 )
             )
      )
    )
  )
)


# Define the server code
server <- function(input, output) {}

# Return a Shiny app object
shinyApp(ui = ui, server = server)