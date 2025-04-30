# Fine-Tuning CNN-LSTM Code with InceptionV3 using Flickr8k + Custom Dataset

import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import sys

# --- Step 1: Load and clean captions ---
def load_doc(filename):
    with open(filename, 'r') as file:
        return file.read()

def parse_captions(doc):
    lines = doc.strip().split('\n')
    captions = {}
    for line in lines:
        parts = line.split(',', 1)
        if len(parts) != 2:
            continue
        img_id = parts[0].strip()
        caption = 'startseq ' + parts[1].strip().lower() + ' endseq'
        if img_id not in captions:
            captions[img_id] = []
        captions[img_id].append(caption)
    return captions

# --- Step 2: Feature extraction using InceptionV3 ---
def build_inception_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    model = Model(base_model.input, base_model.output)
    return model

def extract_features(img_path, model):
    img = keras_image.load_img(img_path, target_size=(299, 299))
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature = model.predict(img, verbose=0)
    return feature

# --- Step 3: Tokenizer and max length ---
def create_tokenizer(captions_dict):
    all_captions = [c for caps in captions_dict.values() for c in caps]
    tokenizer = Tokenizer(oov_token='<unk>')
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

def max_caption_length(captions_dict):
    return min(30, max((len(c.split()) for caps in captions_dict.values() for c in caps), default=0))

# --- GloVe Embedding Loader ---
def load_glove_embeddings(glove_path, tokenizer, embedding_dim=200):
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# --- Step 4: CNN-LSTM Decoder ---
def build_decoder_model(vocab_size, max_length, embedding_matrix=None):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.3)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 200, weights=[embedding_matrix], mask_zero=True, trainable=False)(inputs2)
    se2 = Dropout(0.3)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

# --- Step 5: Data Generator ---
class DataGenerator(Sequence):
    def __init__(self, captions, photos, tokenizer, max_length, vocab_size, batch_size=32):
        self.captions = captions
        self.photos = photos
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.image_ids = [img_id for img_id in captions if img_id in photos and len(captions[img_id]) > 0]

    def __len__(self):
        return max(1, len(self.image_ids) // self.batch_size)

    def __getitem__(self, idx):
        X1, X2, y = [], [], []
        batch_ids = self.image_ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        for img_id in batch_ids:
            photo = self.photos[img_id]
            for caption in self.captions[img_id]:
                seq = self.tokenizer.texts_to_sequences([caption])[0]
                for j in range(1, len(seq)):
                    in_seq, out_seq = seq[:j], seq[j]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length, padding='post')[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    if photo.ndim > 1:
                        photo = np.mean(photo, axis=0)
                    photo = np.mean(photo, axis=0).astype(np.float32) if photo.ndim > 1 else photo.astype(np.float32)
                    X1.append(photo.astype(np.float32))
                    X2.append(in_seq)
                    y.append(out_seq)
        return (np.array(X1, dtype=np.float32), np.array(X2, dtype=np.int32)), np.array(y, dtype=np.float32)

# --- Step 6: Caption Generation (Beam Search) ---
def generate_caption_beam(cnn_model, lstm_model, tokenizer, img_path, max_length, beam_index=5):
    photo = extract_features(img_path, cnn_model)
    start = [tokenizer.word_index['startseq']]
    sequences = [[start, 0.0]]
    while len(sequences[0][0]) < max_length:
        temp = []
        for s in sequences:
            sequence = pad_sequences([s[0]], maxlen=max_length)
            preds = lstm_model.predict([photo, sequence], verbose=0)[0]
            top_preds = np.argsort(preds)[-beam_index:]
            for word in top_preds:
                next_seq = s[0] + [word]
                prob = s[1] + np.log(preds[word] + 1e-10)
                temp.append([next_seq, prob])
        sequences = sorted(temp, key=lambda tup: tup[1], reverse=True)[:beam_index]
    final_seq = sequences[0][0]
    final_caption = [tokenizer.index_word.get(i, '') for i in final_seq if i not in [0, tokenizer.word_index['startseq'], tokenizer.word_index['endseq']]]
    return ' '.join(final_caption)

# --- Step 7: Training / Prediction ---
if __name__ == '__main__':
    if '--predict' in sys.argv:
        print("Prediction Mode")
        tokenizer = pickle.load(open('combined_tokenizer.pkl', 'rb'))
        max_length = 34
        vocab_size = len(tokenizer.word_index) + 1

        embedding_matrix = load_glove_embeddings('glove.6B.200d.txt', tokenizer)
        model = build_decoder_model(vocab_size, max_length, embedding_matrix=embedding_matrix)
        model.load_weights('flickr_custom_combined.weights.h5')

        cnn_encoder = build_inception_model()
        img_path = input("Enter path to image: ").strip()
        if os.path.exists(img_path):
            caption = generate_caption_beam(cnn_encoder, model, tokenizer, img_path, max_length)
            print("\n Caption:", caption)
        else:
            print("Image not found.")

    else:
        print("Training with Flickr8k and custom dataset...")
        doc1 = load_doc('./archive/captions.txt')
        doc2 = load_doc('cleaned_custom_captions.txt')

        captions = parse_captions(doc1)
        captions_custom = parse_captions(doc2)
        captions.update(captions_custom)

        tokenizer = create_tokenizer(captions)
        with open('combined_tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)

        max_length = max_caption_length(captions)
        vocab_size = len(tokenizer.word_index) + 1

        embedding_matrix = load_glove_embeddings('glove.6B.200d.txt', tokenizer)
        model = build_decoder_model(vocab_size, max_length, embedding_matrix=embedding_matrix)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        cnn_encoder = build_inception_model()
        if os.path.exists('combined_features.pkl'):
            with open('combined_features.pkl', 'rb') as f:
                image_features = pickle.load(f)
        else:
            image_features = {}
            from tqdm import tqdm
            for img_id in tqdm(captions, desc='Extracting image features'):
                folder = './custom_images' if os.path.exists(f'./custom_images/{img_id}') else './archive/images'
                img_path = os.path.join(folder, img_id)
                if os.path.exists(img_path):
                    features = extract_features(img_path, cnn_encoder)
                    image_features[img_id] = features
            with open('combined_features.pkl', 'wb') as f:
                pickle.dump(image_features, f)

        img_ids = list(captions.keys())
        train_ids, val_ids = train_test_split(img_ids, test_size=0.1, random_state=42)
        captions_train = {img_id: captions[img_id] for img_id in train_ids}
        captions_val = {img_id: captions[img_id] for img_id in val_ids}

        train_generator = DataGenerator(captions_train, image_features, tokenizer, max_length, vocab_size)
        val_generator = DataGenerator(captions_val, image_features, tokenizer, max_length, vocab_size)

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
        checkpoint = ModelCheckpoint('flickr_custom_combined.weights.h5', monitor='val_loss', save_best_only=True)

        model.fit(train_generator, validation_data=val_generator, epochs=15, callbacks=[checkpoint, early_stop, lr_scheduler])

        from nltk.translate.bleu_score import corpus_bleu
        print("\n Evaluating BLEU score...")
        actual, predicted = [], []
        for img_id in list(captions_val.keys())[:100]:
            if img_id not in image_features:
                continue
            photo = image_features[img_id]
            if photo.ndim == 2:
                photo = np.expand_dims(photo, axis=0)
            folder = './custom_images' if os.path.exists(f'./custom_images/{img_id}') else './archive/images'
            img_path = os.path.join(folder, img_id)
            yhat = generate_caption_beam(cnn_encoder, model, tokenizer, img_path, max_length)
            refs = [cap.split() for cap in captions[img_id]]
            actual.append(refs)
            predicted.append(yhat.split())
        print("BLEU-1:", corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
        print("BLEU-2:", corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
        print("BLEU-3:", corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
        print("BLEU-4:", corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

        # ✅ Manually save weights at the end
        model.save_weights("flickr_custom_combined.weights.h5")
        print("✅ Weights manually saved after training.")
