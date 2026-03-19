library(shiny)
library(plotly)
library(bslib)

parse_datetime_from_filename <- function(x) {
  base <- sub("\\.[Ww][Aa][Vv]$", "", x)
  base <- sub("\\.[Ww][Aa][Vv][Ee]$", "", base)
  as.POSIXct(base, format = "%Y%m%d_%H%M%S", tz = "UTC")
}

normalize_site_name <- function(x) {
  trimws(sub("\\s+[0-9]+$", "", x))
}

load_detections <- function(csv_path) {
  if (!file.exists(csv_path)) {
    stop(sprintf("Archivo no encontrado: %s", csv_path))
  }

  d <- read.csv(csv_path, stringsAsFactors = FALSE)

  if (!all(c("site", "file") %in% names(d))) {
    stop("El CSV debe contener columnas: site, file")
  }

  if (!"site_group" %in% names(d)) {
    d$site_group <- normalize_site_name(d$site)
  }

  d$datetime_utc <- parse_datetime_from_filename(d$file)
  d$datetime_chile <- d$datetime_utc - 3 * 3600
  d$date_chile <- as.Date(d$datetime_chile)
  d$hour_chile <- as.integer(format(d$datetime_chile, "%H"))

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
          div("Selecciona un sitio y filtra por fecha y hora del día.", class = "text-muted")
        ),
        uiOutput("kpi_ui")
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
        sliderInput(
          "hour_range",
          "Rango de horas (Chile)",
          min = 0,
          max = 23,
          value = c(0, 23),
          step = 1
        ),
        checkboxInput("show_points", "Mostrar puntos", value = TRUE),
        div(
          class = "text-muted",
          style = "font-size: 0.9rem;",
          "Tip: Puedes seleccionar un rango que cruza medianoche (por ejemplo 22–6)."
        )
      )
    ),
    column(
      9,
      card(
        card_header("Línea de tiempo (zoom)"),
        plotlyOutput("timeline_plot", height = "640px")
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
    sites <- sort(unique(d$site_group))
    selectInput("site_group", "Sitio", choices = sites, selected = sites[1])
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

    d <- d[d$site_group == input$site_group, , drop = FALSE]

    if (!is.null(input$date_range) && length(input$date_range) == 2) {
      d <- d[d$date_chile >= input$date_range[1] & d$date_chile <= input$date_range[2], , drop = FALSE]
    }

    hr <- input$hour_range
    if (!is.null(hr) && length(hr) == 2) {
      start_hr <- hr[1]
      end_hr <- hr[2]

      if (start_hr <= end_hr) {
        d <- d[d$hour_chile >= start_hr & d$hour_chile <= end_hr, , drop = FALSE]
      } else {
        d <- d[d$hour_chile >= start_hr | d$hour_chile <= end_hr, , drop = FALSE]
      }
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
          "Sitio: ", site_group,
          "<br>Detalle: ", site,
          "<br>Hora (Chile): ", format(datetime_chile, "%Y-%m-%d %H:%M:%S"),
          "<br>Archivo: ", file
        ),
        hoverinfo = "text"
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
          "Sitio: ", site_group,
          "<br>Detalle: ", site,
          "<br>Hora (Chile): ", format(datetime_chile, "%Y-%m-%d %H:%M:%S"),
          "<br>Archivo: ", file
        ),
        hoverinfo = "text"
      )
    }

    plt |>
      layout(
        title = paste0("Detecciones — ", input$site_group),
        xaxis = list(title = "Hora (Chile, UTC-3)", rangeslider = list(visible = TRUE)),
        yaxis = list(title = "", showticklabels = FALSE, zeroline = FALSE)
      ) |>
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
