from TextAlignment.DetailedAnalysis.graph_matching import GraphMatching


class PairwiseComparison:
    def __init__(self,blc, bl_knn,n_gram, min_cliquesize, min_matchsize ):
        self.blc = blc
        self.bl_knn=bl_knn
        self.n_gram=n_gram
        self.min_cliquesize=min_cliquesize
        self.min_matchsize=min_matchsize

    def pairwise_check(self, cf, susp_sents_vect, susp_segms_sent_offset,  src_sents_vect, src_segms_sent_offset,src_lang):
        res_ps =[]
        for seg_pair in cf:
            susp_sent_start=susp_segms_sent_offset[seg_pair[0][0]][0]
            susp_sent_end=susp_segms_sent_offset[seg_pair[0][1]][0]+susp_segms_sent_offset[seg_pair[0][1]][1]
            src_sent_start=src_segms_sent_offset[seg_pair[1][0]][0]
            src_sent_end=src_segms_sent_offset[seg_pair[1][1]][0]+src_segms_sent_offset[seg_pair[1][1]][1]
            cur_res=self.get_match_sents(susp_sents_vect, susp_sent_start, susp_sent_end,src_sents_vect,src_sent_start,src_sent_end, src_lang)
            if cur_res:
                res_ps.append(cur_res)
        return res_ps

    def get_match_sents(self, susp_sents_vect, susp_sent_start, susp_sent_end, src_sents_vect, src_sent_start,src_sent_end, src_lang):
        gm = GraphMatching(self.blc, self.bl_knn, self.n_gram, self.min_cliquesize,self.min_matchsize)
        ps= gm.get_match_segms_pair(susp_sents_vect, susp_sent_start, susp_sent_end, src_sents_vect,src_sent_start,src_sent_end, src_lang)
        return ps


