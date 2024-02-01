from flask import Flask, render_template, request
import torchtext
from function import *
# from function import get_most_similar_word
# from function import Glove_embeddings
app = Flask(__name__)

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
hid_dim = 800
emb_dim = 400
num_layers = 2
dropout_rate = 0.65


device = torch.device('cpu')
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
model = LSTMLanguageModel(vocab_size,hid_dim,emb_dim,num_layers,dropout_rate).to(device)

with open('lstm_lm.pt', 'rb') as f:
    model.load_state_dict(torch.load(f, map_location=device))
model.eval()



max_seq_len = 30
seed = 1234
temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]


@app.route('/', methods=['GET', 'POST'])
def index():
    most_similar_words = None
    if request.method == 'POST':
        search_word = request.form.get('search')
        if search_word:
            most_similar_words = get_generate(search_word, max_seq_len, temperatures, model, tokenizer, vocab, device, seed)
    print(most_similar_words)
    return render_template('index.html', most_similar_words=most_similar_words)
    
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")