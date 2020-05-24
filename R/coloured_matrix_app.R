library(DT)      # for datatable, formatStyle, styleInterval
library(dplyr)   # for %>%

myDT <- matrix(c(-3:2), 3) %>% datatable %>% 
  formatStyle(
    columns = 1:2,
    backgroundColor = styleInterval( 
      cuts = c(-.01, 0), 
      values = c("red", "white", "green")
    )
  )

myDT

library(shiny)
shinyApp(
  ui = fluidPage(DT::dataTableOutput('table')),
  server = function(input, output, session){
    output$table = DT::renderDataTable({myDT})
  }
)