import os
import json

class Num_Word_Exist_Multilingual:
    def __init__(self):
        self.src_segm_based_voc, self.src_cnv_segms_based_voc = {}, {}
        self.total_word=0.0
        self.exist_word=0.0
        self.not_exist=0.0
        self.wordset = []

    def get_statistics(self,filename):
        temp_total_word=0.0
        temp_exist_word=0.0
        temp_not_exist=0.0
        for term,freq in self.src_segm_based_voc.items():
            if term.isalpha() is False :
                continue
            temp_total_word+=1
            if term in self.src_cnv_segms_based_voc:
                temp_exist_word+=1
            else:
                temp_not_exist+=1
        print(filename+'\ttemp_total_word='+str(temp_total_word)+'\ttemp_exist_word='+str(temp_exist_word)+'\ttemp_not_exist='+str(temp_not_exist))
        self.total_word+=temp_total_word
        self.exist_word+=temp_exist_word
        self.not_exist+=temp_not_exist

    def get_stat1istics2(self, filename):
        all_word=0
        temp_total_word = 0.0
        temp_exist_word = 0.0
        temp_not_exist = 0.0
        for term, freq in self.src_segm_based_voc.items():
            if term.isalpha() is False:
                continue
            all_word += 1
            if term in self.wordset:
                continue
            else:
                self.wordset.append(term)
            temp_total_word += 1
            if term in self.src_cnv_segms_based_voc:
                temp_exist_word += 1
            else:
                temp_not_exist += 1
        print(filename + '\tall_word='+str(all_word)+'\ttemp_total_word=' + str(temp_total_word) + '\ttemp_exist_word=' + str(
            temp_exist_word) + '\ttemp_not_exist=' + str(temp_not_exist))
        self.total_word += temp_total_word
        self.exist_word += temp_exist_word
        self.not_exist += temp_not_exist

    def get_stat1istics3(self, filename):
        all_word = 0
        temp_total_word = 0.0
        temp_exist_word = 0.0
        temp_not_exist = 0.0
        for term, freq in self.src_cnv_segms_based_voc.items():
            if term.isalpha() is False:
                continue
            all_word += 1
            # if term in self.wordset:
            #     continue
            # else:
            #     self.wordset.append(term)
            temp_total_word += 1
            if term in self.src_segm_based_voc:
                temp_exist_word += 1
            else:
                temp_not_exist += 1
        print(filename + '\tall_word=' + str(all_word) + '\ttemp_total_word=' + str(
            temp_total_word) + '\ttemp_exist_word=' + str(
            temp_exist_word) + '\ttemp_not_exist=' + str(temp_not_exist))
        self.total_word += temp_total_word
        self.exist_word += temp_exist_word
        self.not_exist += temp_not_exist

    def load_data(self, data_dir, filename):
        d_dir = data_dir + 'src_data/'+ filename + '/'
        f = open(d_dir + 'src_segm_based_voc', 'r')
        self.src_segm_based_voc = json.loads(f.read())
        f = open(d_dir + 'src_cnv_segms_based_voc', 'r')
        self.src_cnv_segms_based_voc = json.loads(f.read())

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

    lang="german"
    # lang="spanish"
    src_lang=""

    da_plag_obj = Num_Word_Exist_Multilingual()
    data_dir = 'C:/Users/Sahelsoft/Desktop/Text Alignment Task/Files/'+benchmark_name+'/'
    da_plag_obj.read_src_lang_file(data_dir)

    num_file=0.0
    filename=''
    for root, dirs, files in os.walk(data_dir+'src_data/'):
        for filename in dirs:
            src_lang = da_plag_obj.src_lang_dict[filename]
            if src_lang != lang:
                continue
            # if filename!="source-document00200.txt":
            #     continue
            num_file+=1
            da_plag_obj.load_data(data_dir, filename)
            da_plag_obj.get_statistics(filename)

    print(benchmark_name + '\t' + '\tlang=' + lang)
    print('wordset='+str(len(da_plag_obj.wordset)))
    print('num_files='+str(num_file))
    print('total_word='+str(da_plag_obj.total_word))
    print('Not exist in embedding='+str(da_plag_obj.exist_word))
    print('exists in dict=' + str(da_plag_obj.not_exist))
    print('percent exist='+str(da_plag_obj.not_exist/da_plag_obj.total_word*100))
