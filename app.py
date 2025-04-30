from flask import Flask, render_template, request
import os
from captioning import build_decoder_model, build_inception_model, generate_caption_beam, load_glove_embeddings
import pickle
import threading
import webbrowser

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model & tokenizer once at app startup
tokenizer = pickle.load(open('combined_tokenizer.pkl', 'rb'))
max_length = 34
vocab_size = len(tokenizer.word_index) + 1

embedding_matrix = load_glove_embeddings('glove.6B.200d.txt', tokenizer)
caption_model = build_decoder_model(vocab_size, max_length, embedding_matrix)
caption_model.load_weights('flickr_custom_combined.weights.h5')
cnn_model = build_inception_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    filename = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            filename = f"uploads/{file.filename}"  # This ensures Flask can access it
            caption = generate_caption_beam(cnn_model, caption_model, tokenizer, filepath, max_length)
    return render_template('index.html', caption=caption, filename=filename)

if __name__ == '__main__':
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=True)
