library(shiny)
library(ggplot2)
library(plotly)
library(bslib)

normalize_site_name <- function(x) {
  trimws(sub("\\s+[0-9]+$", "", x))
}

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
  d$site_base <- normalize_site_name(d$site)

  d
}

ui <- fluidPage(
  theme = bs_theme(version = 5, bootswatch = "flatly"),
  tags$head(
    tags$style(
      HTML(
        "body { padding-top: 16px; }\n"
      )
    )
  ),
  fluidRow(
    column(
      12,
      div(
        style = "display:flex; align-items:baseline; justify-content:space-between; gap: 12px;",
        div(
          h2("Detecciones de embarcaciones", style = "margin-bottom: 0;"),
          div("Explora las detecciones en el tiempo. Usa el zoom para ver detalles.", class = "text-muted")
        ),
        div(
          uiOutput("kpi_ui")
        )
      ),
      tags$hr(style = "margin-top: 12px; margin-bottom: 16px;")
    )
  ),
  fluidRow(
    column(
      3,
      card(
        card_header("Filtros"),
        uiOutput("site_ui"),
        uiOutput("date_ui"),
        checkboxInput("show_points", "Mostrar puntos", value = TRUE),
        div(class = "text-muted", style = "font-size: 0.9rem;",
            "Tip: Arrastra sobre el gráfico para hacer zoom. Doble click para reiniciar.")
      )
    ),
    column(
      9,
      card(
        card_header("Línea de tiempo"),
        plotlyOutput("timeline_plot", height = "620px")
      )
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
    sites <- sort(unique(d$site_base))
    selectInput("site", "Sitio", choices = sites, selected = sites[1])
  })

  output$date_ui <- renderUI({
    d <- detections()
    dateRangeInput(
      "date_range",
      "Rango de fechas (Chile)",
      start = min(d$date_chile, na.rm = TRUE),
      end = max(d$date_chile, na.rm = TRUE),
      min = min(d$date_chile, na.rm = TRUE),
      max = max(d$date_chile, na.rm = TRUE)
    )
  })

  filtered <- reactive({
    d <- detections()

    d <- d[d$site_base == input$site, , drop = FALSE]

    if (!is.null(input$date_range) && length(input$date_range) == 2) {
      d <- d[d$date_chile >= input$date_range[1] & d$date_chile <= input$date_range[2], , drop = FALSE]
    }

    d[order(d$datetime_chile), , drop = FALSE]
  })

  output$kpi_ui <- renderUI({
    d <- filtered()
    n <- nrow(d)

    if (n == 0) {
      return(
        div(
          class = "badge bg-secondary",
          style = "font-size: 0.95rem; padding: 10px 12px;",
          "0 detecciones"
        )
      )
    }

    div(
      class = "badge bg-primary",
      style = "font-size: 0.95rem; padding: 10px 12px;",
      sprintf("%d detecciones", n)
    )
  })

  output$timeline_plot <- renderPlotly({
    d <- filtered()

    if (nrow(d) == 0) {
      return(
        plot_ly() |>
          layout(
            title = "Sin detecciones para los filtros seleccionados",
            xaxis = list(title = "Hora (Chile, UTC-3)"),
            yaxis = list(title = "", showticklabels = FALSE)
          )
      )
    }

    y <- rep(1, nrow(d))

    plt <- if (isTRUE(input$show_points)) {
      plot_ly(
        data = d,
        x = ~datetime_chile,
        y = y,
        type = "scatter",
        mode = "markers",
        marker = list(size = 8, opacity = 0.8),
        text = ~paste0(
          "Archivo: ", file,
          "<br>Hora (Chile): ", format(datetime_chile, "%Y-%m-%d %H:%M:%S"),
          "<br>Sitio: ", site
        ),
        hoverinfo = "text"
      ) |>
        layout(
          title = paste0("Detecciones — ", input$site),
          xaxis = list(title = "Hora (Chile, UTC-3)", rangeslider = list(visible = TRUE)),
          yaxis = list(title = "", showticklabels = FALSE, zeroline = FALSE)
        )
    } else {
      plot_ly(
        data = d,
        x = ~datetime_chile,
        y = y,
        type = "scatter",
        mode = "lines",
        line = list(width = 0),
        text = ~paste0(
          "Archivo: ", file,
          "<br>Hora (Chile): ", format(datetime_chile, "%Y-%m-%d %H:%M:%S"),
          "<br>Sitio: ", site
        ),
        hoverinfo = "text"
      ) |>
        layout(
          title = paste0("Detecciones — ", input$site),
          xaxis = list(title = "Hora (Chile, UTC-3)", rangeslider = list(visible = TRUE)),
          yaxis = list(title = "", showticklabels = FALSE, zeroline = FALSE)
        )
    }

    plt |>
      config(
        displaylogo = FALSE,
        scrollZoom = TRUE,
        modeBarButtonsToRemove = c(
          "select2d",
          "lasso2d",
          "autoScale2d",
          "toggleSpikelines"
        )
      )
  })
}

shinyApp(ui, server)
