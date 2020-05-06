import os
import json

class Avg_Num_Sents:
    def __init__(self):
        self.sents_vect, self.segms_vect = [], []
        self.src_lang_dict={}
        self.num_files=0.0
        self.total_sents=0.0
        self.total_frags=0.0
        self.min_sent=1111111111.0
        self.max_sent =-10.0
        self.min_frag = 1111111111.0
        self.max_frag = -10.0

    def get_statistics(self, filename):
        len_sent_vects=len(self.sents_vect)
        len_frag_vects=len(self.segms_vect)
        print(filename+'\tlen_sent_vects='+str(len_sent_vects)+'\tlen_frag_vects='+str(len_frag_vects))
        self.total_sents += len_sent_vects
        self.total_frags += len_frag_vects
        if len_sent_vects<self.min_sent:
            self.min_sent=len_sent_vects
        if len_sent_vects>self.max_sent:
            self.max_sent=len_sent_vects
        if len_frag_vects<self.min_frag:
            self.min_frag=len_frag_vects
        if len_frag_vects>self.max_frag:
            self.max_frag=len_frag_vects

    def load_data(self, data_dir, filename, type):
        d_dir = data_dir + type+'_data/'+ filename + '/'
        f = open(d_dir + type+'_sents_vect', 'r')
        self.sents_vect = json.loads(f.read())
        f = open(d_dir + type+'_segms_vect', 'r')
        self.segms_vect = json.loads(f.read())


    def read_src_lang_file(self,datadir):
        proj_path = os.path.abspath(os.path.dirname(__file__))
        f = open(data_dir + '/src_lang.txt', 'r+')
        self.src_lang_dict = json.loads(f.read())



'''
MAIN
'''
if __name__ == "__main__":
    # benchmark_name='PAN11'
    # benchmark_name = 'PAN12-Training'
    benchmark_name = 'PAN12-Test'

    # type="susp"
    type="src"

    # lang="english"
    # lang="german"
    # lang="spanish"
    lang="both"

    src_lang=""

    da_plag_obj = Avg_Num_Sents()
    data_dir = 'C:/Users/Sahelsoft/Desktop/Text Alignment Task/Files/'+benchmark_name+'/'
    da_plag_obj.read_src_lang_file(data_dir)

    filename=''
    for root, dirs, files in os.walk(data_dir+type+'_data/'):
        for filename in dirs:
            if type=="src":
                src_lang = da_plag_obj.src_lang_dict[filename]
                if src_lang != lang and lang!="both":
                    continue
            da_plag_obj.num_files+=1
            da_plag_obj.load_data(data_dir, filename,type)
            da_plag_obj.get_statistics(filename)

    print()
    print(benchmark_name+'\t'+'type='+type+'\tlang='+lang)
    print('num_files='+str(da_plag_obj.num_files))
    print('total_sents=' + str(da_plag_obj.total_sents))
    print('avg_sents=' + str(da_plag_obj.total_sents/da_plag_obj.num_files))
    print('min_sents='+str(da_plag_obj.min_sent))
    print('max_sents=' + str(da_plag_obj.max_sent))
    print('total_frags=' + str(da_plag_obj.total_frags))
    print('avg_frags=' + str(da_plag_obj.total_frags/da_plag_obj.num_files))
    print('min_frags='+str(da_plag_obj.min_frag))
    print('max_frags=' + str(da_plag_obj.max_frag))





