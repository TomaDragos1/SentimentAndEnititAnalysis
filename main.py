from flask import Flask, request, render_template, url_for
import sqlite3

import plot_both
import predict_entity
import predict_sentiment

# Initialize the Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    # Create a connection to the database
    conn = sqlite3.connect('reviews.db')
    # Create the database if it doesn't exist
    conn.execute('CREATE TABLE IF NOT EXISTS text_table (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT)')
    # Insert the review into the database
    conn.execute('INSERT INTO text_table (text) VALUES (?)', (review,))
    # Save the changes
    conn.commit()
    # Query the table
    conn.execute('SELECT * FROM text_table')
    # Close the connection
    conn.close()

    # Use the review to generate predictions using your functions
    sentiment_prediction = predict_sentiment.predict(review)
    entity_prediction = predict_entity.predict(review)

    # Create the plots
    plot = plot_both.plot(sentiment_prediction, entity_prediction)
    # Save the figure
    plot.savefig("static/plots.png")

    # Render the images
    return render_template("predictions.html",
                           plot_path=url_for('static', filename='plots.png'))


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
