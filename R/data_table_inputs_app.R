library(shiny)
library(DT)

options(DT.options = list(pageLength = 9))

grid_cell_color_filter <- c(1,1,1,0,0,0,1,1,1)

dt_output = function(title, id) {
  fluidRow(column(
    12, h1(paste0('Table ', sub('.*?([0-9]+)$', '\\1', id), ': ', title)),
    hr(), DTOutput(id)
  ))
}
render_dt = function(data, editable = 'cell', server = TRUE, ...) {

  data <- as.data.frame(cbind(data,
                      grid_cell_color_filter))
  
  data <- DT::datatable(data,
                        rownames = NULL, 
                        colnames = NULL,
                        editable = editable,
                        options = list(dom = 't',
                                       bSort=FALSE,
                                       autoWidth = TRUE,
                                       columnDefs = list(
                                         list(visible=FALSE,targets=c(9)),
                                         list(className = "dt-center", targets = "_all")
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
  
  renderDataTable(data, selection = 'none', server = server)
}

shinyApp(
  ui = fluidPage(
    title = 'Double-click to edit table cells',
    
    dt_output('client-side processing (editable = "cell")', 'x1')
  ),
  
  server = function(input, output, session) {
    d1 = iris
    d1$Date = Sys.time() + seq_len(nrow(d1))
    
    d1 = as.data.frame(matrix(rep(1,81),ncol=9,nrow=9))
    
    # client-side processing
    output$x1 = render_dt(d1, 'cell', FALSE)
    
  }
)