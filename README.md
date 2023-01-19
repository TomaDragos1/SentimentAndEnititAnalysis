
# Sentiment and entity analysis of a review

This project is a web application that takes a user's review as input and predicts
the sentiment and the entity of a review using machine learning. It also includes
a feature that generates plots based on the certainty of the results. The application
is build using the Flask web framework and the machine learning model is built using
Keras with Tenserflow as the backend.



## Requirements

* Python 3
* Flask
* Keras
* Tenserflow
* Pandas
* Numpy
* sqlite3
* Matplotlib
## Usage

To run the app, navigate to the root directory
and run this command:

```
flask run
```
This will start a server at `http://localhost:5000/`.

## Templates and plots
The app has 2 main pages using templates for rendering
HTML:
* `index.html`: The home page, where you can enter the review
* `predictions.html`: The prediction page, where you can see the plots of the prediction.

The plots are made using matplotlib in the plot_both.py. They are
saved in the static folder as plots.png.

## Data

The app uses 2 datasets and 1 database.
* train_dataset.csv
* test_dataset.csv
* reviews.db

The datasets are created using csv_parse{1, 2, 3}.py in the first_database directory
and some datasets from Kaggle:
* Apps: https://www.kaggle.com/datasets/prakharrathi25/google-play-store-reviews
* Places: https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews
* Movies: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
* Products: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews?resource=download

The database stores the reviews given by the users.
## Model

The application uses a pre-trained machine learning model for making
predictions. The model is a Sequential model with an Embedding layer,
an LSTM layer, and a Dense layer. The model is trained using the
`train_dataset.csv` and `test_dataset.csv` (with a split of 4 to 1)
and is saved in the project as best_sent_model.hdf5 and
best_ent_model.hdf5.
You can see some graphs in the `training_validation` directory, where
the accuracy, test_accuracy, loss and test_loss are compared.
## Conclusion

This application is a simple example of how a machine learning
model can be integrated into a web application. The app can be
further developed and improved by adding more features, such as
allowing the user to upload their own dataset. Additionally, the
app could be improved by adding more pre-proccesing layers and a
larger database.