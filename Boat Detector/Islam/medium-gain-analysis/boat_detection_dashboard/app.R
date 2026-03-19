library(shiny)
library(ggplot2)

parse_datetime_from_filename <- function(x) {
  base <- sub("\\.[Ww][Aa][Vv]$", "", x)
  base <- sub("\\.[Ww][Aa][Vv][Ee]$", "", base)
  as.POSIXct(base, format = "%Y%m%d_%H%M%S", tz = "UTC")
}

load_detections <- function(csv_path) {
  if (!file.exists(csv_path)) {
    stop(sprintf("Detections file not found: %s", csv_path))
  }

  d <- read.csv(csv_path, stringsAsFactors = FALSE)

  if (!all(c("site", "file") %in% names(d))) {
    stop("Detections CSV must contain columns: site, file")
  }

  d$datetime_utc <- parse_datetime_from_filename(d$file)
  d$datetime_chile <- d$datetime_utc - 3 * 3600
  d$date_chile <- as.Date(d$datetime_chile)

  d
}

ui <- fluidPage(
  titlePanel("Boat detections timeline"),
  sidebarLayout(
    sidebarPanel(
      uiOutput("site_ui"),
      uiOutput("date_ui"),
      sliderInput(
        "min_tonality",
        "Min tonality",
        min = 0,
        max = 200,
        value = 0,
        step = 1
      ),
      checkboxInput("show_points", "Show points", value = TRUE),
      checkboxInput("show_daily", "Show daily counts", value = TRUE)
    ),
    mainPanel(
      plotOutput("timeline_plot", height = "320px"),
      conditionalPanel(
        condition = "input.show_daily == true",
        plotOutput("daily_plot", height = "240px")
      ),
      tableOutput("summary")
    )
  )
)

server <- function(input, output, session) {
  detections_path <- file.path(getwd(), "outputs", "boat_detections_FINAL.csv")

  detections <- reactive({
    load_detections(detections_path)
  })

  output$site_ui <- renderUI({
    d <- detections()
    selectInput("site", "Site", choices = sort(unique(d$site)), selected = sort(unique(d$site))[1])
  })

  output$date_ui <- renderUI({
    d <- detections()
    dateRangeInput(
      "date_range",
      "Date range (Chile)",
      start = min(d$date_chile, na.rm = TRUE),
      end = max(d$date_chile, na.rm = TRUE),
      min = min(d$date_chile, na.rm = TRUE),
      max = max(d$date_chile, na.rm = TRUE)
    )
  })

  filtered <- reactive({
    d <- detections()

    d <- d[d$site == input$site, , drop = FALSE]

    if (!is.null(input$date_range) && length(input$date_range) == 2) {
      d <- d[d$date_chile >= input$date_range[1] & d$date_chile <= input$date_range[2], , drop = FALSE]
    }

    if ("boat_tonality" %in% names(d)) {
      d <- d[is.na(d$boat_tonality) | d$boat_tonality >= input$min_tonality, , drop = FALSE]
    }

    d[order(d$datetime_chile), , drop = FALSE]
  })

  output$timeline_plot <- renderPlot({
    d <- filtered()

    if (nrow(d) == 0) {
      plot.new()
      text(0.5, 0.5, "No detections for current filters")
      return(invisible(NULL))
    }

    d$y <- 1

    p <- ggplot(d, aes(x = datetime_chile, y = y)) +
      theme_minimal(base_size = 12) +
      labs(
        x = "Time (Chile, UTC-3)",
        y = NULL,
        title = paste("Detections timeline —", input$site)
      ) +
      scale_y_continuous(breaks = NULL)

    if (isTRUE(input$show_points)) {
      if ("boat_tonality" %in% names(d)) {
        p <- p + geom_point(aes(color = boat_tonality), alpha = 0.8, size = 2) +
          scale_color_viridis_c(option = "C", name = "Tonality")
      } else {
        p <- p + geom_point(alpha = 0.8, size = 2)
      }
    } else {
      p <- p + geom_rug(sides = "b")
    }

    p
  })

  output$daily_plot <- renderPlot({
    req(input$show_daily)
    d <- filtered()

    if (nrow(d) == 0) {
      plot.new()
      text(0.5, 0.5, "No detections for current filters")
      return(invisible(NULL))
    }

    daily <- aggregate(list(detections = d$file), by = list(date = d$date_chile), FUN = length)

    ggplot(daily, aes(x = date, y = detections)) +
      geom_col(fill = "#2C7FB8") +
      theme_minimal(base_size = 12) +
      labs(
        x = "Date (Chile)",
        y = "Detections",
        title = "Daily detection counts"
      )
  })

  output$summary <- renderTable({
    d <- filtered()

    if (nrow(d) == 0) {
      return(data.frame(metric = c("site", "detections"), value = c(input$site, 0)))
    }

    out <- data.frame(
      metric = c(
        "site",
        "detections",
        "first_detection_chile",
        "last_detection_chile"
      ),
      value = c(
        input$site,
        nrow(d),
        format(min(d$datetime_chile, na.rm = TRUE), "%Y-%m-%d %H:%M:%S"),
        format(max(d$datetime_chile, na.rm = TRUE), "%Y-%m-%d %H:%M:%S")
      ),
      stringsAsFactors = FALSE
    )

    out
  })
}

shinyApp(ui, server)
