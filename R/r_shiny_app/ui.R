#Define UI for miles per gallon app ----
library(shiny)
library(shinythemes)
library(shinyMatrix)

ui <- navbarPage(
  "Sudoku Solver",
  theme = shinytheme('darkly'),
  
  # Main tab panel for inputting sudoku data and solving puzzle
  tabPanel(
    "Sudoku",
    sidebarLayout(
      sidebarPanel(
        width = 1,
        actionButton('solve','Solve!'),
        h3(''),
        actionButton('clear','Clear')
      ),
      mainPanel(
        width = 11,
        fluidRow(
          column(width = 5,
                 h3('Inputs'),
          matrixInput('initial_grid',
                      value = matrix(data = 1:81,
                                     nrow = 9,
                                     ncol = 9))),
          column(width = 1,
                  h3('')),
          column(width = 5,
                 h3('Solution'))
        )
      )
    )
  ),
  tabPanel(
    'About'#,
    #includeMarkdown('about.md')
  )
)
