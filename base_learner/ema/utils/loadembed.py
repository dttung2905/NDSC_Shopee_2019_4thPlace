
from gensim.models import KeyedVectors
import gc
def load_glove(path, max_features)
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(path))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix_1 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix_1[i] = embedding_vector

    del embeddings_index; gc.collect()
    return embedding_matrix_1

def load_wiki(path, max_features):
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(path) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix_2 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix_2[i] = embedding_vector

    del embeddings_index; gc.collect()
    return embedding_matrix_2


def load_paragram(path,max_features):
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open( EMBEDDING_FILE,encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix_3 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix_3[i] = embedding_vector

    del embeddings_index; gc.collect()
    return embedding_matrix_3




def load_googlenews(path,embedding):
    embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix_4 = (np.random.rand(nb_words, embed_size) - 0.5) / 5.0
    for word, i in word_index.items():
        if i >= max_features: continue
        if word in embeddings_index:
            embedding_vector = embeddings_index.get_vector(word)
            embedding_matrix_4[i] = embedding_vector

    del embeddings_index; gc.collect()
    return embedding_matrix_4


if __name__ =='__main__':
    print('running loadembed.py')
