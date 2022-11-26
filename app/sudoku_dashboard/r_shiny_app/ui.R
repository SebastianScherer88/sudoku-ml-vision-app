#Define UI for miles per gallon app ----
library(shiny)
library(shinythemes)
library(shinyMatrix)
library(DT)

ui <- navbarPage(
  "Sudoku Solver",
  theme = shinytheme('darkly'),
  
  # Main tab panel for inputting sudoku data and solving puzzle
  tabPanel(
    "Sudoku",
    sidebarLayout(
      sidebarPanel(
        width = 1,
        actionButton('clue','Clue'),
        h3(''),
        actionButton('solve','Solve'),
        h3(''),
        actionButton('clear','Clear')
      ),
      mainPanel(
        width = 10,
        fluidRow(
          column(width = 4,
                 h3('Inputs'),
                 h5('Please enter the initial values (1-9) of the sudoku you want to solve, or provide a picture of the sudoku grid.'),
                 matrixInput('initial_grid',
                             value = matrix(data = NA,
                                            nrow = 9,
                                            ncol = 9),
                             rows = list(names = FALSE),
                             cols = list(names = FALSE),
                             class = 'character'),
                 fileInput('initial_picture_upload','Upload picture')),
          column(width = 1,
                 h3('')),
          column(width = 3,
                 h3('Solution'),
                 h5(textOutput('solution_message')),
                 dataTableOutput('solution_grid'))
        )
      )
    )
  ),
  tabPanel(
    'About',
    includeMarkdown('about.md')
  )
)
