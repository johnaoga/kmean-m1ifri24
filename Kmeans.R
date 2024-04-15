library(shiny)
library(ggplot2)
library(DT)  #pour les tableaux

# Définition de l'interface utilisateur (UI)
ui <- fluidPage(
  titlePanel("Clustering K-means avec Shiny"),
  sidebarLayout(
    sidebarPanel(
      numericInput("clusters", "Nombre de clusters :", value = 3, min = 1, max = 10),
      actionButton("runButton", "Exécuter le clustering"),
      numericInput("sepalLength", "Longueur du Sépale :", value = 5.0, min = 0, max = 10, step = 0.1),
      numericInput("sepalWidth", "Largeur du Sépale :", value = 3.0, min = 0, max = 10, step = 0.1),
      verbatimTextOutput("prediction")
    ),
    mainPanel(
      plotOutput("kmeansPlot"),
      dataTableOutput("clusterTable")  # Utilisation de la sortie dataTableOutput
    )
  )
)

# Définition du serveur
server <- function(input, output, session) {
  
  # Fonction pour effectuer le clustering K-means
  kmeans_clustering <- eventReactive(input$runButton, {
    set.seed(123)
    kmeans_result <- kmeans(iris[, 1:4], centers = input$clusters)
    return(kmeans_result)
  })
  
  # Affichage du graphique de dispersion avec les clusters
  output$kmeansPlot <- renderPlot({
    kmeans_result <- kmeans_clustering()
    iris_with_clusters <- cbind(iris, Cluster = as.factor(kmeans_result$cluster))
    ggplot(iris_with_clusters, aes(x = Sepal.Length, y = Sepal.Width, color = Cluster)) +
      geom_point(size = 3) +
      labs(title = "Clustering K-means avec Shiny",
           x = "Longueur du Sépale",
           y = "Largeur du Sépale") +
      theme_minimal()
  })
  
  # Calculer des  statistiques pour chaque cluster et affichage  dans un tableau avec la librairie DT
  output$clusterTable <- renderDataTable({  # Utilisation de renderDataTable
    kmeans_result <- kmeans_clustering()
    cluster_centers <- kmeans_result$centers
    cluster_std <- apply(kmeans_result$centers, 1, sd)  #  Ecart type pour chaque colonne
    cluster_stats <- cbind(cluster_centers, cluster_std)  # Combinez les centroïdes et les écart-types
    colnames(cluster_stats) <- c(colnames(iris[, 1:4]), "Écart type")  # Renommer les colonnes
    return(cluster_stats)
  })
  
  # Prédire le groupe d'un point en fonction des valeurs fournies par l'utilisateur
  output$prediction <- renderText({
    req(input$runButton)  # S'assurer que le clustering a été exécuté
    kmeans_result <- kmeans_clustering()
    
    # Point à prédire
    new_point <- c(input$sepalLength, input$sepalWidth)
    
    # la distance euclidienne entre le nouveau point et les centres de cluster
    distances <- apply(kmeans_result$centers, 1, function(center) sqrt(sum((center - new_point)^2)))
    
    # Index du centre de cluster le plus proche
    predicted_cluster <- which.min(distances)
    
    return(paste("Le groupe prédit pour le point (Longueur du Sépale =", input$sepalLength, ", Largeur du Sépale =", input$sepalWidth, ") est :", predicted_cluster))
  })
}

# Lancer l'application Shiny
shinyApp(ui = ui, server = server)
