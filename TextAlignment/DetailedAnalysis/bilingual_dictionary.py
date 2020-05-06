import io
import numpy as np

class BilingualDictionary:
    def __init__(self,syn_minscore, near_minscore, n_max=110000):
        print('Bilingual models loading:', end=' ')
        # english_dict_path='C:\\Users\\Sahelsoft\\Desktop\\Text Alignment Task\\files\\dictionaries\\'+'english.wiki.multi.vec'
        # self.en_embeddings, self.en_id2word, self.en_word2id,self.en_val_word2id, self.en_tgt_npliang = self.load_vec(english_dict_path, n_max)
        # german_dict_path = 'C:\\Users\\Sahelsoft\\Desktop\\Text Alignment Task\\files\\dictionaries\\' + 'german.wiki.multi.vec'
        # self.de_embeddings, self.de_id2word, self.de_word2id, self.de_val_word2id, self.de_tgt_npliang = self.load_vec(german_dict_path, n_max)
        # spanish_dict_path = 'C:\\Users\\Sahelsoft\\Desktop\\Text Alignment Task\\files\\dictionaries\\' + 'spanish.wiki.multi.vec'
        # self.es_embeddings, self.es_id2word, self.es_word2id,self.es_val_word2id, self.es_tgt_npliang = self.load_vec(spanish_dict_path, n_max)
        self.syn_set={}
        self.near_set={}
        self.syn_minscore=syn_minscore
        self.near_minscore = near_minscore
        print('Done')


    def load_vec(self, emb_path, nmax=170000):
        vectors = []
        word2id = {}
        with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            next(f)
            for i, line in enumerate(f):
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                vectors.append(vect)
                word2id[word] = len(word2id)
                if len(word2id) == nmax:
                    break
        id2word = {v: k for k, v in word2id.items()}
        embeddings = np.vstack(vectors)
        tgt_np_linalg=(embeddings / np.linalg.norm(embeddings, 2, 1)[:, None])
        val_word2id=word2id = {v: k for k, v in id2word.items()}
        return embeddings, id2word, word2id, val_word2id,tgt_np_linalg

    def get_nearest_token(self, word, lang):
        trans=''
        try:
            if lang=='german':
                word_emb = self.de_embeddings[self.de_word2id[word]]
            elif lang=='spanish':
                word_emb = self.es_embeddings[self.es_word2id[word]]
            else:
                word_emb = self.en_embeddings[self.en_word2id[word]]
            scores = self.en_tgt_npliang.dot(word_emb / np.linalg.norm(word_emb))
            gr = scores.argmax()
            trans = self.en_id2word[gr]
            return trans
        except:
            return trans

    def get_synonyms(self, word, lang,k_nn=4):
        # if lang == 'english':
        #     emb = self.en_embeddings
        #     id2word = self.en_id2word
        #     val_word2id=self.en_val_word2id
        #     emp_npliang=self.en_tgt_npliang
        # elif lang == 'german':
        #     emb = self.de_embeddings
        #     id2word = self.de_id2word
        #     val_word2id = self.de_val_word2id
        #     emp_npliang = self.de_tgt_npliang
        # elif lang == 'spanish':
        #     emb = self.es_embeddings
        #     id2word = self.es_id2word
        #     val_word2id = self.es_val_word2id
        #     emp_npliang = self.es_tgt_npliang
        # dict1 = {}
        # try:
        #     word_emb = emb[val_word2id[word]]
        #     scores = emp_npliang.dot(word_emb / np.linalg.norm(word_emb))
        #     k_best = scores.argsort()[-k_nn:][::-1]
        #     for i, idx in enumerate(k_best):
        #         if scores[idx] > 0.3:
        #             dict1[id2word[idx]] = scores[idx]
        #     self.syn_set[word]=dict1
        #     return dict1
        # except:
        #     return dict1
        dict1={}
        if word in self.syn_set:
            for token, score in self.syn_set[word].items():
                # if score > 0.6:
                if score > self.syn_minscore:
                    dict1[token]=score
        return dict1



    def get_nearest_neighbors(self, word, susp_lang, src_lang, k_nn=4):
        # src_emb = self.en_embeddings
        # src_val_word2id=self.en_val_word2id
        # if src_lang == 'german':
        #     tgt_nplinag=self.de_tgt_npliang
        #     tgt_id2word = self.de_id2word
        # elif src_lang == 'spanish':
        #     tgt_nplinag=self.es_tgt_npliang
        #     tgt_id2word = self.es_id2word
        # dict1 = {}
        # try:
        #     word_emb = src_emb[src_val_word2id[word]]
        #     scores = tgt_nplinag.dot(word_emb / np.linalg.norm(word_emb))
        #     k_best = scores.argsort()[-k_nn:][::-1]
        #     for i, idx in enumerate(k_best):
        #         if scores[idx]>0.3:
        #             dict1[tgt_id2word[idx]] = scores[idx]
        #     self.near_set[word]=dict1
        #     return dict1
        # except:
        #     return dict1
        dict1={}
        if word in self.near_set:
            for token, score in self.near_set[word].items():
                if score > self.near_minscore:
                    dict1[token]=score
        return dict1
