import math
import copy

import collections
from operator import itemgetter

class CandidateFragmentIdentification:
    def __init__(self, th1, th2):
        self.th1 = th1
        self.th2 = th2
        # self.numfrag=0.0
        # self.numdoc=0.0
        # self.totalpossiblefrag=0.0

    def sum_vect(self, dic1, dic2):
        res = copy.deepcopy(dic1)
        for i in dic2.keys():
            if i in res:
                res[i] += dic2[i]
            else:
                res[i] = dic2[i]
        return res

    def tf_idf(self, list_dic1, voc1, list_dic2, voc2):
        df = self.sum_vect(voc1, voc2)
        td = len(list_dic1) + len(list_dic2)
        for i in range(len(list_dic1)):
            for j in list_dic1[i].keys():
                list_dic1[i][j] *= math.log(td / float(df[j]))
        for i in range(len(list_dic2)):
            for j in list_dic2[i].keys():
                list_dic2[i][j] *= math.log(td / float(df[j]))

    def eucl_norm(self, d1):
        norm = 0.0
        for val in d1.values():
            norm += float(val * val)
        return math.sqrt(norm)

    def cosine_measure(self, d1, d2):
        dot_prod = 0.0
        det = self.eucl_norm(d1) * self.eucl_norm(d2)
        if det == 0:
            return 0
        for word in d1.keys():
            if word in d2:
                dot_prod += d1[word] * d2[word]
        return dot_prod / det

    def dice_coeff(self, d1, d2):
        if len(d1) + len(d2) == 0:
            return 0
        intj = 0
        for i in d1.keys():
            if i in d2:
                intj += 1
        return 2 * intj / float(len(d1) + len(d2))

    def complete(self, cf, sc):
        res = {}
        for pair, score in cf.items():
            i = pair[0]
            while i >= 0:
                flag = 0
                j = pair[1] - abs(i - pair[0]) + 1
                if j==len(sc[0]):
                    j=len(sc[0])-1
                while j >= 0:
                    if sc[i][j] > self.th2:
                        flag = 1
                        if (i, j) not in res:
                            res[(i, j)] = sc[i][j]
                    else:
                        if abs(j - pair[1] - 1) > abs(i - pair[0]):
                            break
                    j += -1

                if flag == 0:
                    break
                i += -1

            i = pair[0]
            numrows = len(sc)
            while i < numrows:
                flag = 0
                j = pair[1] + abs(i - pair[0]) - 1
                if j<0:
                    j=0
                numcols = len(sc[0])
                while j < numcols:
                    if sc[i][j] > self.th2:
                        flag = 1
                        if (i, j) not in res:
                            res[(i, j)] = sc[i][j]
                    else:
                        if abs(j - pair[1] + 1) > abs(i - pair[0]):
                            break
                    j += 1

                if flag == 0:
                    break
                i += 1
        return res


    def expansion(self, cf):
        ps=[]
        ncf = collections.OrderedDict(sorted(cf.items()))
        cur_pair=[]
        for key, value in ncf.items():
            if cur_pair==[]:
                cur_pair.append(key)
                continue
            if (cur_pair[-1][0] - key[0]) in {1,0,-1} and  (abs(cur_pair[-1][1] - key[1])) <=10:
                cur_pair.append(key)
            else:
                ps.append(cur_pair)
                cur_pair=[]
                cur_pair.append(key)
        if cur_pair!=[]:
            ps.append(cur_pair)
        return ps

    def combine(self, ps):
        res=[]
        rem=[]
        for p_ind in range(0, len(ps)):
            if p_ind not in rem:
                cur_pair=ps[p_ind]
            else:
                continue
            for n_ind in range(p_ind+1,len(ps)):
                for lst_p in cur_pair:
                    if n_ind in rem:
                        continue
                    for lst_n in ps[n_ind]:
                        if (lst_p[0] - lst_n[0]) in {1, 0, -1} and (abs(lst_p[1] - lst_n[1])) <= 10:
                            cur_pair.extend(ps[n_ind])
                            rem.append(n_ind)
                            break
            if cur_pair:
                res.append(cur_pair)
        return res

    def combine_new(self, ps):
        res=[]
        for p_ind in range(0, len(ps)):
            add=[]
            cur_pair=[]
            cur_pair.extend(ps[p_ind])
            for n_ind in range(0,len(res)):
                k=1
                for tup_p in ps[p_ind]:
                    if k==0:
                        break
                    for tup_n in res[n_ind]:
                        if (tup_p[0] - tup_n[0]) in {1, 0, -1} and (abs(tup_p[1] - tup_n[1])) <= 10:
                            add.append(n_ind)
                            cur_pair.extend(res[n_ind])
                            k=0
                            break
            if len(cur_pair)==len(ps[p_ind]):
                res.append(cur_pair)
            else:
                for i in sorted(add,reverse=True):
                    del(res[i])
                res.append(cur_pair)
        return res


    def get_ps_bound(self,cb):
        res=[]
        for lst in cb:
            minX=min(lst, key=itemgetter(0))[0]
            maxX=max(lst,key=itemgetter(0))[0]
            minY=min(lst, key=itemgetter(1))[1]
            maxY = max(lst, key=itemgetter(1))[1]
            res.append([(minX,maxX),(minY,maxY)])
        return res

    def get_risk_CF(self, sc):
        cf={}
        src_len,susp_len=len(sc[0]), len(sc)
        for c in range (0, susp_len):
            for r in range(0,src_len):
                if sc[c][r]>self.th2:
                    cf[(c, r)] = sc[c][r]
        return cf

    def get_candidate_fragment(self, susp_vect, susp_voc, src_vect, src_voc):
        self.tf_idf(susp_vect, susp_voc, src_vect, src_voc)

        w, h = len(src_vect), len(susp_vect)
        sc = [[0.0 for x in range(w)] for y in range(h)]

        cf = {}
        for c in range(len(susp_vect)):
            for r in range(len(src_vect)):
                v1 = self.cosine_measure(susp_vect[c], src_vect[r])
                sc[c][r] = v1
                if v1 > self.th1:
                    cf[(c, r)] = v1
        if cf=={}:
            cf=self.get_risk_CF(sc)
        res_comp= self.complete(cf, sc)
        # self.numdoc+=1
        # self.numfrag+=len(res_comp)
        # self.totalpossiblefrag+=w*h
        # print("numdoc="+str(self.numdoc)+"\tnumfrag="+str(self.numfrag)+"\tnewcompl="+str(len(res_comp))+"\tpossiblefrag="+str(w*h))
        # print("================================================\n")
        res_exp=self.expansion(res_comp)
        res_combine=self.combine_new(res_exp)
        res_bound=self.get_ps_bound(res_combine)
        return res_bound
