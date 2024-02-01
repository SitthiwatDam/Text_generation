# A2 Text Generation

This project is part of building a language model using a text dataset of developer choice. The
objective is to train a model that can generate coherent and contextually relevant text based on a given
input. The project also involves creating a web interface for the search engine.


## Quick Start with Docker Compose

1. **Clone the repository:**
    ```bash
    git clone https://github.com/SitthiwatDam/A2_Text_generation.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd A2_Text_generation
    ```

3. **Build and run the Docker containers:**
    ```bash
    docker-compose up -d
    ```

4. **Access the application:**
    - Open your web browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

<!-- 5. **Submit a search:**
    - Enter a word in the text area.
    - Click the "Submit" button. -->

6. **Stop the application when done:**
    ```bash
    docker-compose down
    ```


## Web application with Flask
This web application is built using Flask, a popular web framework in Python. Leveraging the power of Flask, the application integrates a pre-trained LSTM-LM (Long Short-Term Memory Language Model) from the training phase. The saved model is imported and combined with additional functions within the application. 
![Web application interface](./a2.png)



