$(document).ready(function () {
  $.ajax({
    url: "backend/models",
    type: "GET",
    success: function (models) {
      var pcaModel = models.find((model) => model.name === "PCA");
      if (pcaModel) {
        $("#modelInfo").text("Clustering Model: " + pcaModel.name);
      } else {
        $("#modelInfo").text("No clustering models available.");
      }
    },
  });

  fetch("http://localhost:8000/data_and_plot/")
    .then((response) => response.json())
    .then((data) => {
      var base64Plot = data.plot_base64;

      var img = document.createElement("img");
      img.src = "data:image/png;base64," + base64Plot;

      $("#plot-container").html("");
      $("#plot-container").append(img);
    })
    .catch((error) => {
      console.error("Error fetching plot:", error);
      $("#plot-container").html(
        "<p>Error fetching plot. Please try again later.</p>"
      );
    });

    
});
