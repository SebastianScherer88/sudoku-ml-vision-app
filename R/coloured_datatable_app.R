library(shinyMatrix)
library(shinythemes)
library(shiny)
library(DT)
library(magrittr)

# Global variables can go here
n <- 200
options(DT.options = list(pageLength = 9))
df = as.data.frame(cbind(matrix(as.integer(rnorm(81)), ncol=9),c(1,1,1,0,0,0,1,1,1)))

ui <- fluidPage(
  theme = shinytheme('darkly'),
  column(4,dataTableOutput('table'))
)

server <- function(input, output) {
  output$table <- DT::renderDataTable({
    # style V1 based on values of V6
    datatable(df,
              rownames = NULL, 
              colnames = NULL, 
              options = list(dom = 't',
                             bSort=FALSE,
                             autoWidth = TRUE,
                             columnDefs = list(
                               list(width = '200px', targets = "_all"),
                               list(visible=FALSE,targets=c(9)) # references 10th column - java script/pythonic? indexing starting at 0
                               )
                             ) 
              ) %>% 
      formatStyle(
      c('V1','V2','V3','V7','V8','V9'),'V10',
      backgroundColor = styleEqual(c(0, 1), c('gray', 'white'))
    ) %>% 
      formatStyle(
        c('V4','V5','V6'),'V10',
        backgroundColor = styleEqual(c(1, 0), c('gray', 'white'))
      ) %>%
      formatStyle(
        c('V1','V2','V3','V4','V5','V6','V7','V8','V9'),
        `border-right` = '1px solid black',
        `border-top` = '1px solid black'
        ) %>%
      formatStyle(
        'V1','V10', `border-left` = '1px solid black'
      )
  })
}

# Return a Shiny app object
shinyApp(ui = ui, server = server)