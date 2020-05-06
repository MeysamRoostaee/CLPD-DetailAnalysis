import codecs
import json
import os
import time
import xml.dom.minidom
import xml.etree.ElementTree as ET

from TextAlignment.SimilarityRanking.bilingual_dictionary import BilingualDictionary
from TextAlignment.SimilarityRanking.candidate_fragment_identification import CandidateFragmentIdentification
# from TextAlignment.SimilarityRanking.pairwise_comparison import PairwiseComparison
from TextAlignment.SimilarityRanking.preprocessing_tasks import PreprocessingTasks


class Detailed_Analysis:
    def __init__(self, parameters):
        self.th1 = parameters['th1']
        # self.th1 = 0.1
        self.th2 = parameters['th2']
        # self.th2 = 0.05
        self.min_sentlen = parameters['min_sentlen']
        self.min_plaglen = parameters['min_plaglen']
        self.rssent = parameters['rssent']
        self.rem_sw = parameters['rem_sw']
        self.let_stemming = parameters['let_stemming']
        self.seg_size=parameters['seg_size']
        self.min_segsize=parameters['min_segsize']
        self.fs_knword=int(parameters['fs_knword'])
        self.bl_knn= parameters['bl_knn']
        self.n_gram= parameters['n_gram']
        self.min_cliquesize= parameters['min_cliquesize']
        self.min_matchsize = parameters['min_matchsize']
        self.syn_minscore= parameters['syn_minscore']
        self.near_minscore= parameters['near_minscore']

        self.blc = BilingualDictionary(self.syn_minscore, self.near_minscore, n_max=170000)
        self.pt = PreprocessingTasks(self.blc)
        self.cfi = CandidateFragmentIdentification(self.th1, self.th2)
        # self.pc=PairwiseComparison(self.blc, self.bl_knn, self.n_gram, self.min_cliquesize,self.min_matchsize)

        self.nulldet=0

        self.susp_text, self.susp_lang, self.susp_sents_text, self.susp_sents_vect, self.susp_sents_offset \
            , self.susp_sents_start_end, self.susp_sent_based_voc, self.susp_segms_text, self.susp_segms_vect \
            , self.susp_segms_sent_offset, self.susp_segms_char_offset, self.susp_segm_based_voc \
            = '', "english", [], [], [], [], {}, [], [], [], [], {}
        self.src_text, self.src_lang, self.src_sents_text, self.src_sents_vect, self.src_sents_offset \
            , self.src_sents_start_end, self.src_sent_based_voc, self.src_segms_text, self.src_segms_vect \
            , self.src_segms_sent_offset, self.src_segms_char_offset, self.src_segm_based_voc \
            , self.src_cnv_segms_vect, self.src_cnv_segms_based_voc = '', '', [], [], [], [], {}, [], [], [], [], {}, [], {}
        self.detections = None
        self.cf=None

    def process(self,RatK, susp_filename, susp_text, src_filename, src_text, src_lang):

        # self.preprocess()
        self.compare(RatK)
        self.postprocess()


    def preprocess(self):
        self.susp_sents_text, self.susp_sents_vect, self.susp_sents_offset, self.susp_sents_start_end, self.susp_sent_based_voc = self.pt.tokenize(self.susp_text, self.susp_lang, self.rem_sw, self.let_stemming)
        self.susp_segms_text, self.susp_segms_vect, self.susp_segms_sent_offset, self.susp_segms_char_offset, self.susp_segm_based_voc = self.pt.segmentation(self.susp_sents_text, self.susp_sents_vect, self.susp_sents_offset, self.susp_sents_start_end,self.seg_size, self.min_segsize, self.min_sentlen)

        self.src_sents_text, self.src_sents_vect, self.src_sents_offset, self.src_sents_start_end, self.src_sent_based_voc = self.pt.tokenize(self.src_text, self.src_lang, self.rem_sw, 0)
        self.src_segms_text, self.src_segms_vect, self.src_segms_sent_offset, self.src_segms_char_offset, self.src_segm_based_voc = self.pt.segmentation(self.src_sents_text, self.src_sents_vect, self.src_sents_offset, self.src_sents_start_end, self.seg_size,self.min_segsize, self.min_sentlen)
        self.src_cnv_segms_vect, self.src_cnv_segms_based_voc = self.pt.convert_src_vector(self.src_segms_vect,self.src_lang, self.rem_sw,self.let_stemming,self.fs_knword)

    def compare(self, RatK):
        self.cf=self.cfi.get_candidate_fragment(RatK, self.susp_segms_vect, self.susp_segm_based_voc, self.src_cnv_segms_vect,self.src_cnv_segms_based_voc)

        # self.detections=self.pc.pairwise_check(self.cf, self.susp_sents_vect, self.susp_segms_sent_offset, self.src_sents_vect, self.src_segms_sent_offset, self.src_lang)

    def postprocess(self):
        pass
        # print()

def read_parameters(addr):
    parameters = {}
    tree = ET.parse(addr)
    root = tree.getroot()
    for child in root:
        if child.find('type').text == 'float':
            value = float(child.find('value').text)
        elif child.find('type').text == 'int':
            value = int(child.find('value').text)
        else:
            value = child.find('value').text
        parameters[child.attrib['name']] = value
    return parameters


def read_document(addr, encoding='utf-8'):
    fid = codecs.open(addr, 'r', encoding)
    text = fid.read()
    fid.close()
    return text


def load_data(da_plag_obj, data_dir, susp_filename, src_filename,cur_line, src_lang):
    d_dir = data_dir+'susp_data/' + susp_filename + '/'
    f = open(d_dir + 'susp_sents_text', 'r')
    da_plag_obj.susp_sents_text = json.loads(f.read())
    f = open(d_dir + 'susp_sents_vect', 'r')
    da_plag_obj.susp_sents_vect = json.loads(f.read())
    f = open(d_dir + 'susp_sents_offset', 'r')
    da_plag_obj.susp_sents_offset = json.loads(f.read())
    f = open(d_dir + 'susp_sents_start_end', 'r')
    da_plag_obj.susp_sents_start_end = json.loads(f.read())
    f = open(d_dir + 'susp_sent_based_voc', 'r')
    da_plag_obj.susp_sent_based_voc = json.loads(f.read())
    f = open(d_dir + 'susp_segms_text', 'r')
    da_plag_obj.susp_segms_text = json.loads(f.read())
    f = open(d_dir + 'susp_segms_vect', 'r')
    da_plag_obj.susp_segms_vect = json.loads(f.read())
    f = open(d_dir + 'susp_segms_sent_offset', 'r')
    da_plag_obj.susp_segms_sent_offset = json.loads(f.read())
    f = open(d_dir + 'susp_segms_char_offset', 'r')
    da_plag_obj.susp_segms_char_offset = json.loads(f.read())
    f = open(d_dir + 'susp_segm_based_voc', 'r')
    da_plag_obj.susp_segm_based_voc = json.loads(f.read())

    d_dir = data_dir + 'src_data/' + src_filename + '/'
    da_plag_obj.src_lang=src_lang
    f = open(d_dir + 'src_sents_text', 'r')
    da_plag_obj.src_sents_text = json.loads(f.read())
    f = open(d_dir + 'src_sents_vect', 'r')
    da_plag_obj.src_sents_vect = json.loads(f.read())
    f = open(d_dir + 'src_sents_offset', 'r')
    da_plag_obj.src_sents_offset = json.loads(f.read())
    f = open(d_dir + 'src_sents_start_end', 'r')
    da_plag_obj.src_sents_start_end = json.loads(f.read())
    f = open(d_dir + 'src_sent_based_voc', 'r')
    da_plag_obj.src_sent_based_voc = json.loads(f.read())
    f = open(d_dir + 'src_segms_text', 'r')
    da_plag_obj.src_segms_text = json.loads(f.read())
    f = open(d_dir + 'src_segms_vect', 'r')
    da_plag_obj.src_segms_vect = json.loads(f.read())
    f = open(d_dir + 'src_segms_sent_offset', 'r')
    da_plag_obj.src_segms_sent_offset = json.loads(f.read())
    f = open(d_dir + 'src_segms_char_offset', 'r')
    da_plag_obj.src_segms_char_offset = json.loads(f.read())
    f = open(d_dir + 'src_segm_based_voc', 'r')
    da_plag_obj.src_segm_based_voc = json.loads(f.read())
    f = open(d_dir + 'src_cnv_segms_vect', 'r')
    da_plag_obj.src_cnv_segms_vect=json.loads(f.read())
    f = open(d_dir + 'src_cnv_segms_based_voc', 'r')
    da_plag_obj.src_cnv_segms_based_voc=json.loads(f.read())

    d_dir = data_dir + 'blc_dict/' + cur_line+ '/'
    f = open(d_dir + 'syn_set', 'r')
    da_plag_obj.blc.syn_set= json.loads(f.read())
    f = open(d_dir + 'near_set', 'r')
    da_plag_obj.blc.near_set = json.loads(f.read())


def read_src_lang_file(datadir):
    proj_path = os.path.abspath(os.path.dirname(__file__))
    f = open(data_dir+'/src_lang.txt', 'r+')
    src_lang_dict = json.loads(f.read())
    return src_lang_dict


def det_cf_pos(cf, susp_segms_char_offset, src_segms_char_offset):
    features=[]
    for lst in cf:
        start_pos_susp=susp_segms_char_offset[lst[0][0]][0]
        end_pos_susp=susp_segms_char_offset[lst[0][1]][0]+susp_segms_char_offset[lst[0][1]][1]
        start_pos_src= src_segms_char_offset[lst[1][0]][0]
        end_pos_src= src_segms_char_offset[lst[1][1]][0] + src_segms_char_offset[lst[1][1]][1]
        features.append(((start_pos_susp,end_pos_susp),(start_pos_src,end_pos_src)))
    return features


def det_ps_pos(detections, susp_sents_start_end, src_sents_start_end):
    features = []
    for lst in detections:
        start_pos_susp = susp_sents_start_end[lst[0][0]][0]
        end_pos_susp = susp_sents_start_end[lst[0][1]][1]
        start_pos_src = src_sents_start_end[lst[1][0]][0]
        end_pos_src = src_sents_start_end[lst[1][1]][1]
        features.append(((start_pos_susp, end_pos_susp), (start_pos_src, end_pos_src)))
        # print("ps_char= "+str(((start_pos_susp, end_pos_susp), (start_pos_src, end_pos_src))))
    # print("************************************************************************************")
    # print("\n")
    return features


def serialize_features(susp, src, features, outdir):
    impl = xml.dom.minidom.getDOMImplementation()
    doc = impl.createDocument(None, 'document', None)
    root = doc.documentElement
    root.setAttribute('reference', susp)
    doc.createElement('feature')
    for f in features:
        feature = doc.createElement('feature')
        feature.setAttribute('name', 'detected-plagiarism')
        feature.setAttribute('this_offset', str(f[0][0]))
        feature.setAttribute('this_length', str(f[0][1] - f[0][0]))
        feature.setAttribute('source_reference', src)
        feature.setAttribute('source_offset', str(f[1][0]))
        feature.setAttribute('source_length', str(f[1][1] - f[1][0]))
        root.appendChild(feature)
    doc.writexml(open(outdir + susp.split('.')[0] + '-' + src.split('.')[0] + '.xml', 'w'), encoding='utf-8', newl='\n')

def load_checked_files(out_path):
    checked_files=[]
    for subdir, dirs, files in os.walk(out_path):
        for file in files:
            checked_files.append(file)
    return checked_files


def save_dict(syn_set, near_set, benchmark_name, pair_name):
    outdir = "C:\\Users\\Sahelsoft\\Desktop\\Text Alignment Task\\Files\\"+benchmark_name+"\\blc_dict\\"
    out_d = os.path.dirname(outdir + pair_name + '\\')
    os.makedirs(out_d)
    f = open(out_d + '\\syn_set', 'w+')
    f.write(json.dumps(syn_set))
    f = open(out_d + '\\near_set', 'w+')
    f.write(json.dumps(near_set))


'''
MAIN
'''
if __name__ == "__main__":
    benchmark_name='PAN11'
    # benchmark_name = 'PAN12-Test'

    # RatK=[1, 5,10,20]
    RatK=[5]
    for r in RatK:
        suspdir ='C:/Users/Sahelsoft/Desktop/Text Alignment Task/Dataset/'+benchmark_name+'/All/susp/'
        srcdir = 'C:/Users/Sahelsoft/Desktop/Text Alignment Task/Dataset/'+benchmark_name+'/All/src/'
        outdir_cf = 'C:/Users/Sahelsoft/Desktop/Text Alignment Task/Dataset/Det/outdir-cf/'+benchmark_name+'/'+str(r)+'/'
        data_dir = 'C:/Users/Sahelsoft/Desktop/Text Alignment Task/Files/'+benchmark_name+'/'
        parameters = read_parameters(data_dir+'settings.xml')
        da_plag_obj = Detailed_Analysis(parameters)
        src_lang_dict = read_src_lang_file(data_dir)
        lines = open(data_dir+'Pairs', 'r').readlines()

        t1 = time.time()
        for line in lines:
            da_plag_obj.blc.syn_set = {}
            da_plag_obj.blc.near_set = {}

            susp_filename, src_filename = line.split()
            print(line)

            susp_text='' # susp_text=read_document(os.path.join(suspdir, susp))
            src_text='' # src_text = read_document(os.path.join(srcdir, src))
            src_lang = src_lang_dict[src_filename]

            cur_line=susp_filename.split('.')[0] + '-' + src_filename.split('.')[0]

            # if susp_filename!="suspicious-document00735.txt" or src_filename!="source-document03375.txt":
            #     continue

            load_data(da_plag_obj, data_dir, susp_filename, src_filename,cur_line, src_lang)
            da_plag_obj.process(r,susp_filename, susp_text, src_filename, src_text, src_lang)

            cf_pos = det_cf_pos(da_plag_obj.cf, da_plag_obj.susp_segms_char_offset, da_plag_obj.src_segms_char_offset)
            serialize_features(susp_filename, src_filename, cf_pos, outdir_cf)

            # ps_pos=det_ps_pos(da_plag_obj.detections, da_plag_obj.susp_sents_start_end, da_plag_obj.src_sents_start_end)
            # serialize_features(susp_filename,src_filename, ps_pos, outdir_ps)

            # save_dict(da_plag_obj.blc.syn_set, da_plag_obj.blc.near_set, benchmark_name,cur_line)
            # checked_files.append(cur_line+ '.xml')

        t2 = time.time()
        print('time='+str(t2 - t1))



