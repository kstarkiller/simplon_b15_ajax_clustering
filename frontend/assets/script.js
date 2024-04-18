$(document).ready(function () {
  let varChoice = document.getElementById("varChoice");
  let mse = document.getElementById("mse");
  let bic = document.getElementById("bic");
  let aic = document.getElementById("aic");

  varChoice.onchange = function () {
    if (varChoice.value == "PCA") {
      $.ajax({
        url: "http://kev-ajax-clustering.westeurope.azurecontainer.io:8000/plot_pca/",
        method: "GET",
        dataType: "json",
        success: function (data) {
          var base64Plot = data.plot_base64;
          var img = document.createElement("img");

          img.src = "data:image/png;base64," + base64Plot;
          if (data.MSE) {
            mse.innerText = "MSE: " + data.MSE;
            bic.innerText = "";
            aic.innerText = "";
          } else if (data.BIC && data.AIC) {
            mse.innerText = "";
            bic.innerText = "BIC: " + data.BIC;
            aic.innerText = "AIC: " + data.AIC;
          }

          $("#plot-container").html("");
          $("#plot-container").append(img);
        },
        error: function (error) {
          console.error("Error fetching plot:", error);
          $("#plot-container").html(
            "<p>Error fetching plot. Please try again later.</p>"
          );
        },
      });
    } else if (varChoice.value == "kmeans") {
      $.ajax({
        url: "http://kev-ajax-clustering.westeurope.azurecontainer.io:8000/plot_income_kmeans/",
        method: "GET",
        dataType: "json",
        success: function (data) {
          var base64Plot = data.plot_base64;
          var img = document.createElement("img");

          img.src = "data:image/png;base64," + base64Plot;
          if (data.MSE) {
            mse.innerText = "MSE: " + data.MSE;
            bic.innerText = "";
            aic.innerText = "";
          } else if (data.BIC && data.AIC) {
            mse.innerText = "";
            bic.innerText = "BIC: " + data.BIC;
            aic.innerText = "AIC: " + data.AIC;
          }

          $("#plot-container").html("");
          $("#plot-container").append(img);
        },
        error: function (error) {
          console.error("Error fetching plot:", error);
          $("#plot-container").html(
            "<p>Error fetching plot. Please try again later.</p>"
          );
        },
      });
    } else if (varChoice.value == "gmm") {
      $.ajax({
        url: "http://kev-ajax-clustering.westeurope.azurecontainer.io:8000/plot_age_gmm/",
        method: "GET",
        dataType: "json",
        success: function (data) {
          var base64Plot = data.plot_base64;
          var img = document.createElement("img");

          img.src = "data:image/png;base64," + base64Plot;
          if (data.MSE) {
            mse.innerText = "MSE: " + data.MSE;
            bic.innerText = "";
            aic.innerText = "";
          } else if (data.BIC && data.AIC) {
            mse.innerText = "";
            bic.innerText = "BIC: " + data.BIC;
            aic.innerText = "AIC: " + data.AIC;
          }

          $("#plot-container").html("");
          $("#plot-container").append(img);
        },
        error: function (error) {
          console.error("Error fetching plot:", error);
          $("#plot-container").html(
            "<p>Error fetching plot. Please try again later.</p>"
          );
        },
      });
    }
  };

  // Trigger the onchange event manually to get the default choice
  varChoice.onchange();
});
