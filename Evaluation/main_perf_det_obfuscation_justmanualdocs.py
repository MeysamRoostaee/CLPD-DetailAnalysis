from __future__ import division
from collections import namedtuple
import getopt
import glob
import math
from numpy import int8 as npint8
from numpy.ma import zeros, sum as npsum
import os
import sys
import unittest
import xml.dom.minidom
import json



class PerformanceDetermination:
    def __init__(self):
        self.TREF, self.TOFF, self.TLEN = 'this_reference', 'this_offset', 'this_length'
        self.SREF, self.SOFF, self.SLEN = 'source_reference', 'source_offset', 'source_length'
        self.EXT = 'is_external'
        self.Annotation = namedtuple('Annotation', [self.TREF, self.TOFF, self.TLEN, self.SREF, self.SOFF, self.SLEN, self.EXT])
        self.TREF, self.TOFF, self.TLEN, self.SREF, self.SOFF, self.SLEN, self.EXT = range(7)
        self.detected_file_list=[]


    def macro_avg_recall_and_precision(self,cases, detections):
        """Returns tuple (rec, prec); the macro-averaged recall and precision of the
           detections in detecting the plagiarism cases"""
        return self.macro_avg_recall(cases, detections), \
               self.macro_avg_precision(cases, detections)

    def micro_avg_recall_and_precision(self,cases, detections):
        """Returns tuple (rec, prec); the micro-averaged recall and precision of the
           detections in detecting the plagiarism cases"""
        if len(cases) == 0 and len(detections) == 0:
            return 1, 1
        if len(cases) == 0 or len(detections) == 0:
            return 0, 0
        num_plagiarized, num_detected, num_plagiarized_detected = 0, 0, 0  # chars
        num_plagiarized += self.count_chars(cases)
        num_detected += self.count_chars(detections)
        print(len(detections))
        detections = self.true_detections(cases, detections)
        print(len(detections))
        num_plagiarized_detected += self.count_chars(detections)
        print(num_plagiarized, num_detected, num_plagiarized_detected)
        rec, prec = 0, 0
        if num_plagiarized > 0:
            rec = num_plagiarized_detected / num_plagiarized
        if num_detected > 0:
            prec = num_plagiarized_detected / num_detected
        return rec, prec

    def granularity(self,cases, detections):
        """Granularity of the detections in detecting the plagiarism cases."""
        if len(detections) == 0:
            return 1
        detections_per_case = list()
        case_index = self.index_annotations(cases, self.TREF)
        det_index = self.index_annotations(detections, self.TREF)
        for tref in case_index:
            cases, detections = case_index[tref], det_index.get(tref, False)
            if not detections:  # No detections for document tref.
                continue
            for case in cases:
                num_dets = sum((self.is_overlapping(case, det) for det in detections))
                detections_per_case.append(num_dets)
        detected_cases = sum((num_dets > 0 for num_dets in detections_per_case))
        if detected_cases == 0:
            return 1
        return sum(detections_per_case) / detected_cases


    def plagdet_score(self, rec, prec, gran):
        """Combines recall, precision, and granularity to a allow for ranking."""
        if (rec == 0 and prec == 0) or prec < 0 or rec < 0 or gran < 1:
            return 0
        return ((2 * rec * prec) / (rec + prec)) / math.log(1 + gran, 2)


    def macro_avg_recall(self, cases, detections):
        """Recall of the detections in detecting plagiarism cases."""
        if len(cases) == 0 and len(detections) == 0:
            return 1
        elif len(cases) == 0 or len(detections) == 0:
            return 0
        num_cases, recall_per_case = len(cases), list()
        case_index = self.index_annotations(cases, self.TREF)
        det_index = self.index_annotations(detections, self.TREF)
        for tref in case_index:
            cases, detections = case_index[tref], det_index.get(tref, False)
            if not detections:  # No detections for document tref.
                continue
            for case in cases:
                recall_per_case.append(self.case_recall(case, detections))
        return sum(recall_per_case) / num_cases


    def case_recall(self, case, detections):
        """Recall of the detections in detecting the plagiarism case."""
        num_detected_plagiarized = self.overlapping_chars(case, detections)
        num_plagiarized = case[self.TLEN] + case[self.SLEN]
        return num_detected_plagiarized / num_plagiarized


    def macro_avg_precision(self, cases, detections):
        """Precision of the detections in detecting the plagiarism cases."""
        # Observe the difference to calling 'macro_avg_recall(cases, detections)'.
        return self.macro_avg_recall(detections, cases)


    def true_detections(self, cases, detections):
        """Recreates the detections so that only true detections remain and so that
           the true detections are reduced to the passages that actually overlap
           with the respective detected case."""
        true_dets = list()
        case_index = self.index_annotations(cases, self.TREF)
        det_index = self.index_annotations(detections, self.TREF)
        for tref in case_index:
            cases, detections = case_index[tref], det_index.get(tref, False)
            if not detections:  # No detections for document tref.
                continue
            for case in cases:
                case_dets = (det for det in detections if self.is_overlapping(case, det))
                true_case_dets = (self.overlap_annotation(case, det) for det in case_dets)
                true_dets.extend(true_case_dets)
        return true_dets


    def overlap_annotation(self, ann1, ann2):
        """Returns an Annotation that annotates overlaps between ann1 and ann2."""
        tref, sref, ext = ann1[self.TREF], ann1[self.SREF], ann1[self.EXT] and ann2[self.EXT]
        toff, tlen, soff, slen = 0, 0, 0, 0
        if self.is_overlapping(ann1, ann2):
           toff, tlen = self.overlap_chars(ann1, ann2, self.TOFF, self.TLEN)
           if ext:
               soff, slen = self.overlap_chars(ann1, ann2, self.SOFF, self.SLEN)
        return self.Annotation(tref, toff, tlen, sref, soff, slen, ext)


    def overlap_chars(self, ann1, ann2, xoff, xlen):
        """Returns the overlapping passage between ann1 and ann2, given the keys
           xoff and xlen."""
        overlap_start, overlap_length = 0, 0
        max_ann = ann1 if ann1[xoff] >= ann2[xoff] else ann2
        min_ann = ann1 if ann1[xoff] < ann2[xoff] else ann2
        if min_ann[xoff] + min_ann[xlen] > max_ann[xoff]:
           overlap_start = max_ann[xoff]
           overlap_end = min(min_ann[xoff] + min_ann[xlen], \
                             max_ann[xoff] + max_ann[xlen])
           overlap_length = overlap_end - overlap_start
        return overlap_start, overlap_length


    def count_chars(self, annotations):
        """Returns the number of chars covered by the annotations, while counting
           overlapping chars only once."""
        num_chars = self.count_chars2(annotations, self.TREF, self.TOFF, self.TLEN)
        num_chars += self.count_chars2(annotations, self.SREF, self.SOFF, self.SLEN)
        return num_chars


    def count_chars2(self, annotations, xref, xoff, xlen):
        """Returns the number of cvhars covered by the annotations with regard to
           the keys xref, xoff, and xlen."""
        num_chars = 0
        max_length = max((ann[xoff] + ann[xlen] for ann in annotations))
        char_bits = zeros(max_length, dtype=bool)
        xref_index = self.index_annotations(annotations, xref)
        for xref in xref_index:
            annotations = xref_index[xref]
            char_bits[:] = False
            for ann in annotations:
                char_bits[ann[xoff]:ann[xoff] + ann[xlen]] = True
            num_chars += npsum(char_bits)
        return num_chars


    def overlapping_chars(self, ann1, annotations):
        """Returns the number of chars in ann1 that overlap with the annotations."""
        annotations = [ann2 for ann2 in annotations if self.is_overlapping(ann1, ann2)]
        if len(annotations) == 0 or not isinstance(ann1, self.Annotation):
            return 0
        this_overlaps = zeros(ann1[self.TLEN], dtype=bool)
        source_overlaps = zeros(ann1[self.SLEN], dtype=bool)
        for ann2 in annotations:
            self.mark_overlapping_chars(this_overlaps, ann1, ann2, self.TOFF, self.TLEN)
            self.mark_overlapping_chars(source_overlaps, ann1, ann2, self.SOFF, self.SLEN)
        return npsum(this_overlaps) + npsum(source_overlaps)


    def mark_overlapping_chars(self,char_bits, ann1, ann2, xoff, xlen):
        """Sets the i-th boolean in char_bits to true if ann2 overlaps with the i-th
           char in ann1, respecting the given xoff and xlen index."""
        offset_difference = ann2[xoff] - ann1[xoff]
        overlap_start = min(max(0, offset_difference), ann1[xlen])
        overlap_end = min(max(0, offset_difference + ann2[xlen]), ann1[xlen])
        char_bits[overlap_start:overlap_end] = True


    def is_overlapping(self, ann1, ann2):
        """Returns true iff the ann2 overlaps with ann1."""
        detected = ann1[self.TREF] == ann2[self.TREF] and \
                   ann2[self.TOFF] + ann2[self.TLEN] > ann1[self.TOFF] and \
                   ann2[self.TOFF] < ann1[self.TOFF] + ann1[self.TLEN]
        if ann1[self.EXT] == True and ann2[self.EXT] == True:
            detected = detected and ann1[self.SREF] == ann2[self.SREF] and \
                       ann2[self.SOFF] + ann2[self.SLEN] > ann1[self.SOFF] and \
                       ann2[self.SOFF] < ann1[self.SOFF] + ann1[self.SLEN]
        return detected


    def index_annotations(self, annotations, xref):
        """Returns an inverted index that maps references to annotation lists."""
        index = dict()
        for ann in annotations:
            index.setdefault(ann[xref], []).append(ann)
        return index


    def extract_annotations_from_file(self, xmlfile, tagname, srclang_list, autoset_list,manual_obfuscation_type):
        """Returns a set of plagiarism annotations from an XML file."""
        doc = xml.dom.minidom.parse(xmlfile)
        annotations = set()
        if not doc.documentElement.hasAttribute('reference'):
            return annotations
        t_ref = doc.documentElement.getAttribute('reference')
        if t_ref in autoset_list:
            return annotations
        for node in doc.documentElement.childNodes:
            if node.nodeType == xml.dom.Node.ELEMENT_NODE and \
               node.hasAttribute('name') and \
               node.getAttribute('name').endswith(tagname):
                ann = self.extract_annotation_from_node(node, t_ref,srclang_list,manual_obfuscation_type)
                if ann:
                    annotations.add(ann)
        return annotations


    def extract_annotations_from_files(self, path, tagname,srclang_list, autoset_list, manual_obfuscation_type):
        """Returns a set of plagiarism annotations from XML files below path."""
        if not os.path.exists(path):
            print("Path not accessible:" + path)
            sys.exit(2)
        annotations = set()
        xmlfiles = glob.glob(os.path.join(path, '*.xml'))
        xmlfiles.extend(glob.glob(os.path.join(path, os.path.join('*', '*.xml'))))
        for xmlfile in xmlfiles:
            annotations.update(self.extract_annotations_from_file(xmlfile, tagname,srclang_list,autoset_list,manual_obfuscation_type))
        return annotations


    def extract_annotation_from_node(self, xmlnode, t_ref,srclang_list, manual_obfuscation_type):
        """Returns a plagiarism annotation from an XML feature tag node."""
        if not (xmlnode.hasAttribute('this_offset') and \
                xmlnode.hasAttribute('this_length')):
            return False
        t_off = int(xmlnode.getAttribute('this_offset'))
        t_len = int(xmlnode.getAttribute('this_length'))
        s_ref, s_off, s_len, ext = '', 0, 0, False
        if xmlnode.hasAttribute('source_reference') and \
           xmlnode.hasAttribute('source_offset') and \
           xmlnode.hasAttribute('source_length'):
            s_ref = xmlnode.getAttribute('source_reference')
            if s_ref not in srclang_list:
                return False
            if xmlnode.hasAttribute('manual_obfuscation'):
                if xmlnode.getAttribute('manual_obfuscation') != manual_obfuscation_type and manual_obfuscation_type!="both":
                    return False
            s_off = int(xmlnode.getAttribute('source_offset'))
            s_len = int(xmlnode.getAttribute('source_length'))
            ext = True
        return self.Annotation(t_ref, t_off, t_len, s_ref, s_off, s_len, ext)

    def usage(self):
        """Prints command line usage manual."""
        """\
    Usage: perfmeasures.py [options]

    Options:
          --micro      Compute micro-averaged recall and precision,
                       default: macro-averaged recall and precision
      -p, --plag-path  Path to the XML files with plagiarism annotations
          --plag-tag   Tag name suffix of plagiarism annotations,
                       default: 'plagiarism'
      -d, --det-path   Path to the XML files with detection annotations
          --det-tag    Tag name of the detection annotations,
                       default: 'detected-plagiarism'
      -h, --help       Show this message
    """


    def read_srclang(self,srclang_path,lang):
        f = open(srclang_path + '\\src_lang.txt', 'r+')
        lang_dict = json.loads(f.read())
        src_lang_list=[]
        for x in lang_dict:
            if lang_dict[x]==lang or lang=="both":
                src_lang_list.append(x)
        return src_lang_list

    def read_autoset(self,autoset_path):
        f = open(autoset_path+ '\\autoset', 'r+')
        auto_dict = json.loads(f.read())
        auto_list=[]
        for x in auto_dict:
            auto_list.append(x)
        return auto_list

    # def parse_options(self, det_path):
    def parse_options(self):
        benchmark_name='PAN11'
        # benchmark_name='PAN12-Test'


        # lang='both'
        lang='german'
        # lang = 'spanish'


        # obfuscation = "automatic"
        obfuscation="manual"
        # obfuscation = "both"

        print("benchmark:====>  "+benchmark_name)
        print("lang:=====>  "+lang)
        print("obfuscation:====>  "+obfuscation)
        """Parses the command line options."""
        try:
            long_options = ["micro", "plag-path=", "plag-tag=", "det-path=",
                            "det-tag=", "help"]
            opts, _ = getopt.getopt(sys.argv[1:], "p:d:h", long_options)
        except getopt.GetoptError as err:
            print(err)
            self.usage()
            sys.exit(2)
        micro_averaged = False
        plag_path = "C:\\Users\\Sahelsoft\\Desktop\\Text Alignment Task\\Dataset\\"+benchmark_name+"\\All\\info\\"
        # det_path = "C:\\Users\\Sahelsoft\\Desktop\\Text Alignment Task\\Dataset\\Det\\outdir-cf\\"+benchmark_name+"\\"
        det_path = "C:\\Users\\Sahelsoft\\Desktop\\Text Alignment Task\\Dataset\\Det\\outdir-ps\\"+benchmark_name+"\\"
        srclang_path="C:\\Users\\Sahelsoft\\Desktop\\Text Alignment Task\\Files\\"+benchmark_name+"\\"
        autoset_path="C:\\Users\\Sahelsoft\\Desktop\\Text Alignment Task\\Files\\"+benchmark_name+"\\"
        plag_tag_name, det_tag_name = "plagiarism", "detected-plagiarism"
        for opt, arg in opts:
            if opt in ("--micro"):
                micro_averaged = True
            elif opt in ("-p", "--plag-path"):
                plag_path = arg
            elif opt == "--plag-tag":
                plag_tag_name = arg
            elif opt in ("-d", "--det-path"):
                det_path = arg
            elif opt == "--det-tag":
                det_tag_name = arg
            elif opt in ("-h", "--help"):
                self.usage()
                sys.exit()
            else:
                assert False, "Unknown option."
        if plag_path == "undefined":
            print("Plagiarism path undefined. Use option -p or --plag-path.")
            sys.exit()
        if det_path == "undefined":
            print("Detections path undefined. Use option -d or --det-path.")
            sys.exit()
        return (micro_averaged, plag_path, plag_tag_name, det_path, det_tag_name, srclang_path,lang, autoset_path, obfuscation)


    def main(self, micro_averaged, plag_path, plag_tag_name, det_path, det_tag_name,srclang_path,lang, autoset_path, obfuscation):
        """Main method of this module."""
        srclang_list=self.read_srclang(srclang_path,lang)
        autoset_list=self.read_autoset(autoset_path)
        print('Reading ' + plag_path)
        if obfuscation=="automatic":
            manual_obfuscation_type="false"
        elif obfuscation=="manual":
            manual_obfuscation_type = "true"
        else:
            manual_obfuscation_type = "both"
        cases = self.extract_annotations_from_files(plag_path, plag_tag_name,srclang_list,autoset_list,manual_obfuscation_type)
        print('Reading ' + det_path)
        detections = self.extract_annotations_from_files(det_path, det_tag_name,srclang_list,autoset_list, manual_obfuscation_type)

        print('Processing... (this may take a while)')
        rec, prec = 0, 0
        if micro_averaged:
            rec, prec = self.micro_avg_recall_and_precision(cases, detections)
        else:
            rec, prec = self.macro_avg_recall_and_precision(cases, detections)
        gran = self.granularity(cases, detections)
        print('Plagdet Score= ' + str(self.plagdet_score(rec, prec, gran)))
        print('Recall= ' + str(rec))
        print('Precision= ' + str(prec))
        print('Granularity= ' + str(gran))
        print()


# benchmark_name='PAN11'
# det_path = "C:\\Users\\Sahelsoft\\Desktop\\Text Alignment Task\\Dataset\\Det\\outdir-ps\\"+benchmark_name+"\\"
# for subdir,dirs,files in os.walk(det_path):
#     for directory in dirs:
#         print(directory)
#         PD=PerformanceDetermination()
#         PD.main(*PD.parse_options(det_path+directory+"\\"))
#         print("===============================")

PD=PerformanceDetermination()
PD.main(*PD.parse_options())