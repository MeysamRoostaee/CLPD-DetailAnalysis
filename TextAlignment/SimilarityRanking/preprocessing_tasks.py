import Stemmer
import copy
import os.path

import nltk


class PreprocessingTasks:
    def __init__(self,blc):
        self.blc=blc
        self.langstopwords=self.read_stopwords()
        print()

    def read_stopwords(self):
        dir_path = "C:\\Users\\Sahelsoft\\Desktop\\Text Alignment Task\\Files\\"
        lang=['english','german','spanish']
        lang_stop_list={}
        for ll in lang:
            file_path='stopwords\\'+ll+'_stopwords.txt'
            path = dir_path+ file_path
            stop_words = []
            lines = open(path, 'r').readlines()
            for line in lines:
                stop_words.append(line.strip())
            lang_stop_list[ll]=stop_words
        return  lang_stop_list

    def tokenize(self, text, lang, rem_sw,let_stemming):
        sents_text, sents_offset, sents_start_end, sent_based_voc= [],[],[],{}
        text = text.replace(chr(0), ' ')
        text = text.replace('*', ' ')
        text = text.replace('(', ' ')
        text = text.replace(')', ' ')
        text = text.replace('|', ' ')
        text = text.replace('\ufeff', ' ')

        sent_detector = nltk.data.load('tokenizers/punkt/'+lang+'.pickle')
        stemmer = Stemmer.Stemmer(lang)
        word_detector = nltk.TreebankWordTokenizer()
        sent_spans = sent_detector.span_tokenize(text)
        if rem_sw == 0:
            stopwords = []
        elif rem_sw == 1:
            stopwords=copy.deepcopy(self.langstopwords[lang])
        sents_vect = []
        for span in sent_spans:  # For each sentence
            sent_dic = {}
            sents_text.append(text[span[0]: span[1]].lower())
            for word in word_detector.tokenize(sents_text[-1]):  # for each word in the sentence
                if len(word) > 2 and word not in stopwords:
                    if let_stemming == 1:
                        word_pp = stemmer.stemWord(word)
                    else:
                        word_pp = word
                else:
                    continue
                if word_pp in sent_dic:
                    sent_dic[word_pp] += 1
                else:
                    sent_dic[word_pp] = 1
                    if word_pp in sent_based_voc:
                        sent_based_voc[word_pp] += 1
                    else:
                        sent_based_voc[word_pp] = 1

            sents_vect.append(sent_dic)
            sents_offset.append([span[0], span[1] - span[0]])
            sents_start_end.append([span[0],span[1]])
        return sents_text, sents_vect, sents_offset, sents_start_end, sent_based_voc

    def segmentation(self,sents_text, sents_vect, sents_offset, sents_start_end, seg_size, min_segsize,min_sentlen):
        segms_text,segms_vect, segms_sent_offset, segms_char_offset, segm_based_voc =[],[],[],[],{}
        temp_segm_text, temp_segm_vec = '', {}
        seg_start=0
        curr_segsize=0;
        i, range_i = 0, len(sents_text)
        while i<range_i:
            temp_segm_text+=sents_text[i]
            for word,freq in sents_vect[i].items():
                if word in temp_segm_vec:
                    temp_segm_vec[word]+=freq
                else:
                    temp_segm_vec[word]=freq
                    if word in segm_based_voc:
                        segm_based_voc[word]+=1
                    else:
                        segm_based_voc[word] = 1
            if sum(sents_vect[i].values())>min_sentlen:
                curr_segsize+=1
            if curr_segsize==seg_size:
                segms_text.append(temp_segm_text)
                segms_vect.append(temp_segm_vec)
                segms_char_offset.append([sents_start_end[seg_start][0], sents_start_end[i][1] - sents_start_end[seg_start][0]])
                segms_sent_offset.append([seg_start,i-seg_start])
                temp_segm_text, temp_segm_vec='', {}
                seg_start=i+1
                curr_segsize=0
            i+=1
        if temp_segm_text!='':
            if curr_segsize>= min_segsize:
                segms_text.append(temp_segm_text)
                segms_vect.append(temp_segm_vec)
                segms_char_offset.append([sents_start_end[seg_start][0], sents_start_end[i-1][1] - sents_start_end[seg_start][0]])
                segms_sent_offset.append([seg_start,i-1-seg_start])
            else:
                segms_text[-1]+=temp_segm_text
                for word,freq in temp_segm_vec.items():
                    if word in segms_vect[-1]:
                        segms_vect[-1][word]+=freq
                        segm_based_voc[word] -= 1
                    else:
                        segms_vect[-1][word] = freq
                segms_char_offset[-1][1]=sents_start_end[i-1][1] - segms_char_offset[-1][0]
                segms_sent_offset[-1][1]=i-1-segms_sent_offset[-1][0]
        return segms_text,segms_vect, segms_sent_offset, segms_char_offset, segm_based_voc

    def convert_src_vector(self, src_vect,lang, rem_sw, let_stemming,fs_knword):
        res_vect, res_voc=[],{}
        if rem_sw == 0:
            stopwords = []
        elif rem_sw == 1:
            stopwords=copy.deepcopy(self.langstopwords['english'])
        stemmer = Stemmer.Stemmer('english')
        for seg_vect in src_vect:
            temp_res_vect={}
            for word, freq in seg_vect.items():
                # trans_w_lst= self.blc.get_nearest_token(word, lang, fs_knword)
                # if trans_w_lst:
                #     for trans_w,score in trans_w_lst.items():
                #         if trans_w not in stopwords:
                #             if let_stemming==0:
                #                 temp_res_vect[trans_w]=freq
                #             else:
                #                 trans_w=stemmer.stemWord(trans_w)
                #                 temp_res_vect[trans_w] = freq
                trans_w = self.blc.get_nearest_token(word, lang, fs_knword)
                if trans_w:
                    if trans_w not in stopwords:
                        if let_stemming == 0:
                            temp_res_vect[trans_w] = freq
                        else:
                            trans_w = stemmer.stemWord(trans_w)
                            temp_res_vect[trans_w] = freq
                else:
                    trans_w = word
                    temp_res_vect[trans_w] = freq
            res_vect.append(temp_res_vect)
            for tw in temp_res_vect.keys():
                if tw in res_voc:
                    res_voc[tw] += 1
                else:
                    res_voc[tw] = 1
        return res_vect,res_voc


# pt=PreprocessingTasks()
# blc = BilingualDictionary(n_max=50000)
# time1=time.time()
# espTexts = 'Zudem hatte ich genug mit mir selbst zu thun, ich hatte mir fest vorgenommen, ins Innere von Marokko zu gehen, um dort im Dienste der Regierung meine medicinischen Kenntnisse zu verwerthen. Zu der Zeit sprach man in Spanien und Algerien viel von einer Reorganisation der marokkanischen Armee; es hiess, der Sultan habe nach dem Friedensschlusse mit Spanien die Absicht ausgesprochen, Reformen einzuführen; man las in den Zeitungen Aufforderungen, nach Marokko zu gehen, jeder Europäer könne dort sein Wissen und sein Können verwerthen. Dies Alles beschäftigte mich, ich machte die schönsten Pläne, ich dachte um so eher in Marokko fortkommen zu können, als ich durch jahrelangen Aufenthalt in Algerien acclimatisirt war; ich glaubte um so eher mich den Verhältnissen des Landes anschmiegen zu können, als ich in Algerien gesucht hatte, mich der arabischen Bevölkerung zu nähern und mit der Sitte und Anschauungsweise dieses Volkes mich bekannt zu machen. Um Mitternacht wurde ein kurzer Halt vor Nemours (Djemma Rassaua) gemacht, um Passagiere abzusetzen und einzunehmen, und wieder ging es weiter nach dem Westen, und als es am folgenden Morgen tagte, befanden wir uns gerade in gleicher Höhe von Melilla. Ich unterlasse es, eine Beschreibung der Küstenfahrt zu geben, von der sich überdies äusserst wenig sagen lässt. Nackt, steil und abschreckend fallen die Felswände ins Meer hinein. Freilich ist die Küste gar nicht so einförmig, wie sie sich in einer Entfernung von circa dreissig Seemeilen ausnimmt, welche Entfernung wir gewöhnlich hielten, auch konnte man deutlich manchmal Wald und Buschwerk unterscheiden; aber das belebende Element fehlt, kein Dorf, kein Städtchen ist zu erblicken, höchstens die einsame Kuppel des Grabmals irgend eines Heiligen sagt dem Vorbeifahrenden, dass auch dort an der Küste Menschen hausen. Hätte nicht Spanien einige befestigte Punkte, Strafanstalten, an dieser Küste, sie würde vollkommen unbewohnt erscheinen. Alhucemas, Pegnon de Velez bekamen wir nach einander von ferne zu sehen, als einzige Zeichen von Menschenbauten. Denn wenn auch die Rifbewohner einige Dörfer an der Küste haben, so sind diese doch so versteckt angelegt, dass sie sich dem Auge des Vorbeifahrenden entziehen. Der Seeräuber scheut das Licht, er muss Schlupfwinkel haben, und die in unmittelbarer Nähe des Mittelmeers wohnenden Rifi sind nichts Anderes als Seeräuber, und zwar der schlimmsten Art.'
# sents_text, sents_vect, sents_offset, sents_start_end, sent_based_voc= pt.tokenize(espTexts,'german',1,0)
# segms_text,segms_vect, segms_sent_offset, segms_char_offset, segm_based_voc =pt.segmentation(sents_text,sents_vect,sents_offset,sents_start_end,5,2,3)
# src_cnv_vect, src_cnv_voc=pt.convert_src_vector(segms_vect,"german",1,1,1)
# susp_vect=copy.deepcopy(src_cnv_vect)
# susp_voc=copy.deepcopy(src_cnv_voc)
# pt.tf_idf(src_cnv_vect,src_cnv_voc,susp_vect,susp_voc)
# for c in range(len(susp_vect)):
#     for r in range(len(src_cnv_vect)):
#         cosine_score=pt.cosine_measure(susp_vect[c],src_cnv_vect[r])
#         dice_coeff_score=pt.dice_coeff(susp_vect[c],src_cnv_vect[r])
#         print("cosine="+str(cosine_score))
#         print("dice="+str(dice_coeff_score))
#     print()
# print()
# time2=time.time()
# print(time2-time1)
