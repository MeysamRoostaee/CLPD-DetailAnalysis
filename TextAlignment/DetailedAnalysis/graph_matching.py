import matplotlib

matplotlib.use('TkAgg')
import networkx as nx


class GraphMatching:
    def __init__(self, blc, bl_knn, n_gram, min_cliquesize, min_matchsize):
        self.blc = blc
        self.bl_knn = bl_knn
        self.n_gram=n_gram
        self.min_cliquesize=min_cliquesize
        self.min_matchsize = min_matchsize
        self.srcrel_graph = nx.Graph()
        self.match_graph = nx.Graph()

    def create_srcrel_graph(self, src_sents_vect, src_sent_start, src_sent_end, lang):
        for sent_index in range(src_sent_start, src_sent_end+1):
            checked_words=[]
            for token in src_sents_vect[sent_index]:
                if self.srcrel_graph.has_node(token):
                    checked_words.append(token)
                    cur_snums = self.srcrel_graph.node[token]['sr_sent']
                    if sent_index not in cur_snums:
                        self.srcrel_graph.node[token]['sr_sent'].append(sent_index)
                else:
                    n_token = self.check_srcrel_graph_has_synnode(token, lang)
                    if n_token:
                        checked_words.append(n_token)
                        cur_snums = self.srcrel_graph.node[n_token]['sr_sent']
                        if sent_index not in cur_snums:
                            self.srcrel_graph.node[n_token]['sr_sent'].append(sent_index)
                    else:
                        self.srcrel_graph.add_node(token,sr_sent=[sent_index])
                        checked_words.append(token)
            for i in range(0, len(checked_words)):
                n=i-self.n_gram+1
                j=i-1
                while j>=0 and j>=n:
                    self.srcrel_graph.add_edge(checked_words[i],checked_words[j])
                    j=j-1


    def create_match_graph(self,susp_sents_vect, susp_sent_start, susp_sent_end, susp_lang, src_lang):
        for sent_index in range(susp_sent_start, susp_sent_end+1):
            checked_words=[]
            i=-1
            for token in susp_sents_vect[sent_index]:
                flag = False
                i+=1
                if self.match_graph.has_node(token):
                    self.match_graph.node[token]['su_sent'].append(sent_index)
                    checked_words.append((token,i))
                    flag = True
                else:
                    n_token = self.check_match_graph_has_synnode(token, susp_lang)
                    if n_token:
                        self.match_graph.node[n_token]['su_sent'].append(sent_index)
                        checked_words.append((n_token, i))
                        flag = True
                    else:
                        n_token=self.check_srcrel_graph_has_crossnode(token, susp_lang, src_lang)
                        if n_token:
                            cur_sr_sent=self.srcrel_graph.node[n_token]['sr_sent']
                            self.match_graph.add_node(token, su_sent=[sent_index],sr_sent=cur_sr_sent)
                            checked_words.append((token, i))
                            flag = True
                if flag == False:
                    checked_words.append(("Non", i))
            for i in range(0, len(checked_words)):
                n=i-self.n_gram+1
                j=i-1
                while j>=0 and j>=n and n<=checked_words[j][1]:
                    if checked_words[j][0] != "Non" and checked_words[i][0] != "Non":
                        self.match_graph.add_edge(checked_words[i],checked_words[j])
                    j=j-1


    def check_srcrel_graph_has_synnode(self, token, lang):
        nn_tokens = self.blc.get_synonyms(token, lang, self.bl_knn)
        for n_token, value in nn_tokens.items():
            if self.srcrel_graph.has_node(n_token):
                return n_token
        return False


    def check_match_graph_has_synnode(self,token, lang):
        nn_tokens = self.blc.get_synonyms(token, lang, self.bl_knn)
        for n_token, value in nn_tokens.items():
            if self.match_graph.has_node(n_token):
                return n_token
        return False


    def check_srcrel_graph_has_crossnode(self,token, susp_lang, src_lang):
        nn_tokens = self.blc.get_nearest_neighbors(token, susp_lang, src_lang, self.bl_knn)
        for n_token, value in nn_tokens.items():
            if self.srcrel_graph.has_node(n_token):
                return n_token
        return False


    def extract_cliques(self):
        susp_sid, src_sid=[],[]
        cliques= [g for g in nx.clique.find_cliques(self.match_graph) if len(g) >= self.min_cliquesize]
        for subgraph in cliques:
            for cnode in subgraph:
                for sent_index in self.match_graph.node[cnode[0]]['su_sent']:
                    if sent_index not in susp_sid:
                        susp_sid.append(sent_index)
                for sent_index in self.match_graph.node[cnode[0]]['sr_sent']:
                    if sent_index not in src_sid:
                        src_sid.append(sent_index)
        print("susp_side=  "+str(sorted(susp_sid)))
        print("src_side=  "+str(sorted(src_sid)))
        print("==================")
        if src_sid and susp_sid:
            return ([(min(susp_sid), max(susp_sid)),(min(src_sid), max(src_sid))])
        else:
            return False

    def new_extract_cliques(self):
        total_susp_sid, total_src_sid=set(), set()
        cliques= [g for g in nx.clique.find_cliques(self.match_graph) if len(g) >= self.min_cliquesize]
        for subgraph in cliques:
            sub_clique_susp_sid, sub_clique_src_sid={},{}
            for cnode in subgraph:
                for sent_index in self.match_graph.node[cnode[0]]['su_sent']:
                    if sent_index not in sub_clique_susp_sid:
                        sub_clique_susp_sid[sent_index]=1
                    else:
                        sub_clique_susp_sid[sent_index]+=1
                for sent_index in self.match_graph.node[cnode[0]]['sr_sent']:
                    if sent_index not in sub_clique_src_sid:
                        sub_clique_src_sid[sent_index]=1
                    else:
                        sub_clique_src_sid[sent_index]+=1
            temp_susp_sid, temp_src_sid =set(), set()
            for key,value in sub_clique_susp_sid.items():
                if value>=self.min_matchsize:
                    temp_susp_sid.add(key)
            for key,value in sub_clique_src_sid.items():
                if value>=self.min_matchsize:
                    temp_src_sid.add(key)
            if temp_susp_sid and temp_src_sid:
                total_susp_sid=total_susp_sid | temp_susp_sid
                total_src_sid=total_src_sid | temp_src_sid
        if total_src_sid and total_susp_sid:
            return ([(min(total_susp_sid), max(total_susp_sid)),(min(total_src_sid), max(total_src_sid))])
        else:
            return False


    def get_match_segms_pair(self, susp_sents_vect, susp_sent_start, susp_sent_end, src_sents_vect,src_sent_start,src_sent_end, src_lang):
        susp_lang = 'english'
        self.create_srcrel_graph(src_sents_vect, src_sent_start, src_sent_end, src_lang)
        self.create_match_graph(susp_sents_vect,susp_sent_start, susp_sent_end, susp_lang, src_lang)
        ps = self.new_extract_cliques()
        return ps
