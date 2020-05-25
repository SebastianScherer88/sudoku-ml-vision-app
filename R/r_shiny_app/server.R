# Define server logic to plot various variables against mpg ----
library(shiny)
library(httr)

# --- some constants
# valid grid cell values for initial sudoku
grid_cell_value_range <- c('1','2','3','4','5','6','7','8','9','')
grid_cell_color_filter <- c(1,1,1,0,0,0,1,1,1)

# solving endpoint of python-based FastAPI RestAPI that does the solving of sudoku via integer programming
sudoku_solver_url <- '127.0.0.1:8000/solve/'

# --- helper functions
# checks if the initial values matrix has valid entries
has_valid_range <- function(grid_matrix){
  grid_vals <- unlist(grid_matrix)
  
  valid_range <- all(grid_vals %in% grid_cell_value_range)
  
  return (valid_range)
}

# resets invalid entries of a given initial values matrix to NULL values
reset_invalid_entries <- function(grid_matrix){
  grid_vals <- unlist(grid_matrix)
  
  invalid_entries <- !grid_vals %in% grid_cell_value_range
  grid_vals [invalid_entries] <- ''
  
  valid_grid_matrix <- matrix(grid_vals,
                              ncol = 9)
  
  return (valid_grid_matrix)
}

# derives a nested list representing initial value constraints compatible to the InitianValueConstraints pydantic model
# expected by the sudoku solver python RestAPI
derive_constraints_from_initial_grid <- function(grid_matrix){
  constraints <- list()
  
  counter <- 1
  
  for (i in 1:nrow(grid_matrix)){
    for (j in 1:ncol(grid_matrix)){
      if (grid_matrix[i,j] != list('')){
        # append constraint list based on pydantic model 'InitialValueConstraint' defined in sudoku_solver_api.py
        constraints[[counter]] <- list(row_index = as.integer(i-1),
                                       column_index = as.integer(j-1),
                                       value = as.integer(grid_matrix[i,j][[1]]))
        
        counter <- counter + 1
      }
    }
  }
  
  # put into list based on pydantic model 'InitialValueConstraints'
  initial_constraints <- list(initial_values = constraints)
  
  return (initial_constraints)
}

server <- function(input, output, session) {
  
  # check user inputs and remove invalid ones
  observeEvent(
    input$initial_grid,
    {
      # sanity check user inputs
      is_valid_range <- has_valid_range(input$initial_grid)
      
      if (!is_valid_range){
        # print dialog box with informative value range message
        showModal(modalDialog(
          title = "Careful!",
          "Initial values must be between 1 and 9 (inclusive) only.",
          easyClose = TRUE,
          footer = NULL
        ))
        
        # undo invalid inputs and updte UI initial value matrix accordingly
        updateMatrixInput(session,
                          'initial_grid',
                          reset_invalid_entries(input$initial_grid))
      }
    }
  )
  
  # reset entire initial value grid if required
  observeEvent(
    input$clear,
    {
      # undo invalid inputs and updte UI initial value matrix accordingly
      updateMatrixInput(session,
                        'initial_grid',
                        matrix(data = NA,
                               nrow = 9,
                               ncol = 9))
    }
  )
  
  # try and solve current initial value grid by calling out to sudoku solver python RestAPI
  initial_constraints <- eventReactive(
    input$solve,
    {
      derive_constraints_from_initial_grid(input$initial_grid)
    }
  )
  
  sudoku_solver_response <- reactive(
    {
      input$solve
      response <- content(POST(sudoku_solver_url,
                               body = initial_constraints(), encode = "json"))
    }
  )
  
  solution_grid <- reactiveVal(value = NULL)
  solution_message <- reactiveVal(value = NULL)
  
  observeEvent(
    input$solve,
    {
      if (sudoku_solver_response()$solved == 1){
        # render solved sudoku
        solution_grid(do.call(rbind,sudoku_solver_response()$solution))
        solution_message('Sudoku with given initial values was successfully solved :)')
      } else if (sudoku_solver_response()$solved == -1){
        # render empty grid to highlight lack of solution
        #solution_grid(matrix(data = NA, nrow = 9,ncol = 9))
        solution_grid(NULL)
        solution_message('Sudoku with given initial values was not solvable :(')
        
        # print dialog box with informative value range message
        showModal(modalDialog(
          title = "Unlucky!",
          "Please check whether you have accidentally put multiple identical digits in the same row/column/box. \n
          Even if that isnt the case, if too many/the wrong initial values are given, the resulting constraints create a Sudoku that isnt solvable. \n
          In that case, try some other initial values.",
          easyClose = TRUE,
          footer = NULL
        ))
      }
      
      print(solution_grid())
    }
        )
  
  observeEvent(
    input$clear,
    {
      # render empty grid to highlight lack of solution
      solution_grid(NULL)
      solution_message(NULL)
      
      print(solution_grid())
    }
  )
  
  augmented_solution_grid <- reactive({})
  
  output$solution_grid <- DT::renderDataTable({
    
    augmented_solution_grid <- as.data.frame(cbind(solution_grid(),
                                                   grid_cell_color_filter))
    
    if (is.null(solution_grid())){
      datatable(augmented_solution_grid,
                options = list(dom = 't',bSort=FALSE,columnDefs = list(list(visible=FALSE, targets = "_all"))))
    } else if (!is.null(solution_grid())){
      
      datatable(augmented_solution_grid,
                rownames = NULL, 
                colnames = NULL, 
                options = list(dom = 't',
                               bSort=FALSE,
                               autoWidth = TRUE,
                               columnDefs = list(
                                 list(className = "dt-center", targets = "_all"),
                                 list(width = '200px', targets = "_all"),
                                 list(visible=FALSE,targets=c(9)) # references 10th column - java script/pythonic? indexing starting at 0
                               )
                )
      ) %>% 
        formatStyle(
          columns = 1:10, color = 'white'
        ) %>%
        formatStyle(
          c('V1','V2','V3','V7','V8','V9'),'grid_cell_color_filter',
          backgroundColor = styleInterval(c(0.5), c('black','rgb(90,90,90)'))
        ) %>% 
        formatStyle(
          c('V4','V5','V6'),'grid_cell_color_filter',
          backgroundColor = styleInterval(c(0.5), c('rgb(90,90,90)','black'))
        ) %>%
        formatStyle(
          c('V1','V2','V3','V4','V5','V6','V7','V8','V9'),
          `border-right` = '1px solid white',
          `border-top` = '1px solid white',
          `border-bottom` = '1px solid white'
        ) %>%
        formatStyle(
          'V1','grid_cell_color_filter', `border-left` = '1px solid white'
        )
    }
  })
  
  output$solution_message <- renderText({solution_message()})
    }