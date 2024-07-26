# my_feature_extractor
from msilib.schema import Feature
import re
from collections import defaultdict, Counter
from socket import IPV6_CHECKSUM, getfqdn
from tkinter import ALL, N
from uuid import UUID
from bs4 import FeatureNotFound

import numpy
from scipy import stats
from joblib.parallel import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer

import warnings
import pandas as pd
import time

from copy import deepcopy

warnings.filterwarnings("ignore")

RANDOM_STATE = None
LIST_FEATURES = ["_parts", "_n_grams"]
    # _parts 分为了四列的独热
    # _n_grams 分为了7组，每组是一个统计量，组内分3列，每列是相应n-gram

def clean_domain_list(domain_list: list, dga=False):
    """对域名列表进行初始化操作，包括去首尾空字符、去空值、对DGA域名的格式处理（可能针对特殊DGA域名信息格式）
    
    TODO: 修改适配
    Cleans a given domain list from invalid domains and cleans each single domain in the list.
    :param domain_list:
    :param dga:
    :return:
    """

    domain_list = [d.strip() for d in domain_list]   # 小写域名 但对我们的不能全部小写，DNS tunneling包含大写字母
    domain_list = list(filter(None, domain_list))  # 过滤None值

    if dga:
        # some ramnit domains ending with the pattern: [u'.bid', u'.eu']  # 蠕虫病毒
        to_remove = []
        for d in domain_list:
            if '[' in d:
                print('Domain contains [: {!s}'.format(d))
                to_remove.append(d)
                res = set()
                bracket_split = d.split('[')
                tlds = bracket_split[1].split(',')
                for tld in tlds:
                    tld = tld.strip()
                    tld = tld.split("'")[1].replace('.', '')
                    res_d = bracket_split[0] + tld
                    res.add(res_d)
                    print('Cleaned domain: {!s}'.format(res_d))
                    domain_list.append(res_d)

        domain_list = [d for d in domain_list if d not in to_remove]

    return domain_list


class PublicSuffixes:
    """常见的Mozilla维护的官方后缀类
    
    Represents the official public suffixes list maintained by Mozilla  https://publicsuffix.org/list/
    """
    def __init__(self, file='./feature/public_suffix.txt'):
        """读取公开后缀列表 Public Suffix List
        公开后缀的列表包括：TLD、公司或组织提交的公开后缀
        Args:
            file (str, optional): 文件路径. Defaults to './feature/public_suffix.txt'.
        """        
        with open(file, encoding='utf-8') as f:
            self.data = f.readlines()

        self.data = clean_domain_list(self.data)
        self.data = ['.' + s for s in self.data if not (s.startswith('/') or s.startswith('*'))]
        # print(self.data)
        self.data = clean_domain_list(self.data)  # 得到常见后缀名


    def get_valid_tlds(self):
        """得到有效的TLD

        Returns:
            list: 有效的2级TLD列表
        """        
        return [s for s in self.data if len(s.split('.')) == 2]

    def get_valid_public_suffixes(self):  # 得到有效域名后缀
        """得到有效的域名后缀

        Returns:
            list: 过滤后的有效域名列表
        """        
        return self.data
    
class Keywords:
    def __init__(self, file='./feature/keywords.txt'):
        self.parameter_keywords = set()
        self.normal_keywords = set()
        self.ptr = self.parameter_keywords
        with open(file, encoding='utf-8') as f:
            for each_line in f:
                if each_line.startswith('#') and "parameter" in each_line:
                    self.ptr = self.parameter_keywords
                elif each_line.startswith('#') and "normal" in each_line:
                    self.ptr = self.normal_keywords
                elif not len(each_line.strip()) == 0:
                    self.ptr.add(each_line.strip())
    
    def get_parameter_keywords(self):
        return [kw for kw in self.parameter_keywords]
    
    def get_normal_keywords(self):
        return [kw for kw in self.normal_keywords]
    
class UnsuitableFeatureOrderException(Exception):
    pass

# 正则表达式部分
ipv4_pattern = re.compile("(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])(\.|\-)){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])")  # 符合1.1.1.1或1-1-1-1格式
ipv6_pattern = re.compile("^((([0-9A-Fa-f]{1,4}(\.|\-)){7}[0-9A-Fa-f]{1,4})|(([0-9A-Fa-f]{1,4}(\.|\-)){1,7}(\.|\-))|(([0-9A-Fa-f]{1,4}(\.|\-)){6}(\.|\-)[0-9A-Fa-f]{1,4})|(([0-9A-Fa-f]{1,4}(\.|\-)){5}((\.|\-)[0-9A-Fa-f]{1,4}){1,2})|(([0-9A-Fa-f]{1,4}(\.|\-)){4}((\.|\-)[0-9A-Fa-f]{1,4}){1,3})|(([0-9A-Fa-f]{1,4}(\.|\-)){3}((\.|\-)[0-9A-Fa-f]{1,4}){1,4})|(([0-9A-Fa-f]{1,4}(\.|\-)){2}((\.|\-)[0-9A-Fa-f]{1,4}){1,5})|([0-9A-Fa-f]{1,4}(\.|\-)((\.|\-)[0-9A-Fa-f]{1,4}){1,6})|((\.|\-)((\.|\-)[0-9A-Fa-f]{1,4}){1,7})|(([0-9A-Fa-f]{1,4}(\.|\-)){6}(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])(\\.(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])){3})|(([0-9A-Fa-f]{1,4}(\.|\-)){5}(\.|\-)(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])(\\.(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])){3})|(([0-9A-Fa-f]{1,4}(\.|\-)){4}((\.|\-)[0-9A-Fa-f]{1,4}){0,1}(\.|\-)(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])(\\.(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])){3})|(([0-9A-Fa-f]{1,4}(\.|\-)){3}((\.|\-)[0-9A-Fa-f]{1,4}){0,2}(\.|\-)(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])(\\.(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])){3})|(([0-9A-Fa-f]{1,4}(\.|\-)){2}((\.|\-)[0-9A-Fa-f]{1,4}){0,3}(\.|\-)(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])(\\.(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])){3})|([0-9A-Fa-f]{1,4}(\.|\-)((\.|\-)[0-9A-Fa-f]{1,4}){0,4}(\.|\-)(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])(\\.(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])){3})|((\.|\-)((\.|\-)[0-9A-Fa-f]{1,4}){0,5}(\.|\-)(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])(\\.(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])){3}))$")
hex_32_pattern = re.compile("[0-9A-Fa-f]{32}")
uuid_pattern = re.compile("[a-fA-F0-9]{8}(\-|\.)?[a-fA-F0-9]{4}(\-|\.)?[a-fA-F0-9]{4}(\-|\.)?[a-fA-F0-9]{4}(\-|\.)?[a-fA-F0-9]{12}")
reuuid_pattern = re.compile("[a-fA-F0-9]{12}(\-|\.)?[a-fA-F0-9]{4}(\-|\.)?[a-fA-F0-9]{4}(\-|\.)?[a-fA-F0-9]{4}(\-|\.)?[a-fA-F0-9]{8}")

HEX_DIGITS = set('0123456789abcdef') # 十六进制数
VOWELS = set('aeiou') # 元音
PARTS_MAX_CONSIDERED = 4 # 最大考虑部分，与子域位置的独热特征相关

PUBLIC_SUF = PublicSuffixes()  # 公共后缀集类
VALID_TLDS = PUBLIC_SUF.get_valid_tlds()  # TLD source https://publicsuffix.org/ 有效顶级域名尾 ******但是python有自带的库能自动分割sld，之后可以替换
VALID_PUB_SUFFIXES = PUBLIC_SUF.get_valid_public_suffixes()  # 有效公共后缀

KEYWORDS = Keywords()  # 关键词类
PARAMETER_KEYWORDS = KEYWORDS.get_parameter_keywords()  # 参数子域名相关关键词
NORMAL_KEYWORDS = KEYWORDS.get_normal_keywords()  # 正常子域名相关关键词

NOW_TIMESTAMP = time.time()

__domain = ''  # 域名
__dot_split = ()  # 无点分域名
__joined_dot_split = ''  # 无点域名
__dot_split_suffix_free = ()  # 无点分无后缀域名
__joined_dot_split_suffix_free = ''  # 无点无后缀域名
__public_suffix = ''  # 公共后缀
__unigram = ()  # 1-gram


def extract_features(d: str, features):
    """对给定的特征集提取特征
    
    Extract all features given as arguments from the given domain
    :param features: arbitrary many features, names of the public functions
    :param d: domain name as string
    :param debug: set to true, to return a tuple of a scaled and unscaled feature vector
    :return: scaled feature vector according to input data (in case debug is set to True: feature_vector, scaled_vector)
    """

    __fill_cache(d)

    feature_vector = [d]

    # print('Extracting features for: {!s}'.format(d))
    # print('Cache: {!s}\n{!s}\n{!s}\n{!s}\n{!s}'.format(__dot_split, __joined_dot_split, __dot_split_suffix_free,
                                                    # __joined_dot_split_suffix_free, __public_suffix))

    # workaround for defect domain data in the benign set (actually should have been filtered out, but one always forget cases)
    # sadly not working, kept here for later analysis maybe
    
    if len(__joined_dot_split_suffix_free) == 0:
        print('Defect domain with non-TLD length of zero: {!s}'.format(d))
        return [0] * len(features)
    
    # using exception here for more robustness due to defect data + performance better than if else statements
    for f in features:
        try:
            feature_vector = feature_vector + f()
            # print(f())
        except (ValueError, ArithmeticError) as e:
            # XXX maybe better approach than setting to zero?
            # log.error('Feature {!s} could not be extracted of {!s}. Setting feature to zero'.format(f, d))
            feature_vector = feature_vector + [0]

    # 所有信息打印
    # print('\n{!s}, {!s}, {!s}'.format(d, [f.__name__ for f in features], feature_vector))

    return feature_vector


def __fill_cache(domain: str):
    """填充当前域名的各种形式

    Args:
        domain (str): 原始域名
    """    
    global __dot_split, __joined_dot_split, __dot_split_suffix_free, __joined_dot_split_suffix_free, __public_suffix, __domain
    __domain = domain  # ab.c.baidu.com
    __dot_split = tuple(domain.split('.'))  # (ab, c, baidu, com)
    __joined_dot_split = ''.join(list(__dot_split))  # abcbaiducom
    __dot_split_suffix_free, __public_suffix = __public_suffix_remover(__dot_split)  # (ab, c) baidu.com
    __joined_dot_split_suffix_free = ''.join(__dot_split_suffix_free)  # abc


def __public_suffix_remover(dot_split):
    """最长公共后缀移除函数，如果没有找到，则返回原域名
    
    Finds the largest matching public suffix
    :param dot_split: 
    :return: public suffix free domain as dot split, public suffix
    """
    match = ''

    if len(dot_split) < 2:
        return tuple(dot_split), match

    for i in range(0, len(dot_split)):
        sliced_domain_parts = dot_split[i:]
        match = '.' + '.'.join(sliced_domain_parts)
        if match in VALID_PUB_SUFFIXES:  # baidu.com
            cleared = dot_split[0:i]  # ab.c
            return tuple(cleared), match
    return tuple(dot_split), match


def _vowel_ratio():
    """元音率：如果有字母则返回元音/字母数，否则返回0
    
    Ratio of vowels to non-vowels
    :return: vowel ratio
    """
    vowel_count = 0
    alpha_count = 0
    domain = __joined_dot_split_suffix_free
    for c in domain:
        if c in VOWELS:
            vowel_count += 1
        if c.isalpha():
            alpha_count += 1

    if alpha_count > 0:
        return [vowel_count/alpha_count]
    else:
        return [0]


def _digit_ratio():
    """数字率：数字在域名长度（无点无后缀）中占的比例
    
    Determine ratio of digits to domain length
    :return:
    """
    domain = __joined_dot_split_suffix_free
    digit_count = 0
    for c in domain:
        if c.isdigit():
            digit_count += 1

    return [digit_count/len(domain)]


def _length():
    """域名长度
    
    Determine domain length
    :return:
    """
    return [len(__domain)]


def _contains_wwwdot():
    """是否存在www.
    
    1 if 'www. is contained' 0 else
    :return:
    """
    if 'www.' in __domain:
        return [1]
    else:
        return [0]


def _contains_subdomain_of_only_digits():   # 帮助标签
    """检查是否存在只有数字的域名
    
    Checks if subdomains of only digits are contained.
    :return: 
    """
    for p in __dot_split:
        only_digits = True
        for c in p:
            if c.isalpha():
                only_digits = False
                break
        if only_digits:
            return [1]
    return [0]


def _subdomain_lengths_mean():
    """子域名长度均值
    
    Calculates average subdomain length
    :return:
    """
    overall_len = 0
    for p in __dot_split_suffix_free:
        overall_len += len(p)
    return [overall_len / len(__dot_split_suffix_free)]


def _parts():
    """以独热形式返回去除熟知域名尾后的最长部分到哪一个，特征数量（独热编码长度）取决于PARTS_MAX_CONSIDERED
    
    Calculate the number of domain levels present in a domain, where rwth-aachen.de evaluates to 1 -> [1,0,0,0,0]
    The feature is decoded in a binary categorical way in the form [0,0,0,1,0]. The index represents the number of subdomains
    :return:
    """

    feature = [0] * PARTS_MAX_CONSIDERED
    split_length = len(__dot_split_suffix_free)
    if split_length >= PARTS_MAX_CONSIDERED:
        feature[PARTS_MAX_CONSIDERED - 1] = 1
    else:
        feature[split_length - 1] = 1

    return [feature]  # 修改


# def _contains_ip_addr():


def _contains_digits():
    """是否包含数字

    Returns:
        list: 域名中任何时候出现数字，返回1，否则返回0
    """
    if any(char.isdigit() for char in __domain):
        return [1]
    else:
        return [0]


def _has_valid_tld():
    """是否包含有效的顶级域名
    
    Checks if the domain ends with a valid TLD
    :return:
    """
    if __public_suffix:
        return [1]
    return [0]


def _contains_one_char_subdomains():
    """检查是否包含只有一个字符的子域名
    
    Checks if the domain contains subdomains of only one character
    :return:
    """
    parts = __dot_split

    if len(parts) > 2:
        parts = parts[:-1]

    for p in parts:
        if len(p) == 1:
            return [1]

    return [0]


def _prefix_repetition():
    """前缀是否在之后的部分独立重复
    
    Checks if the string is prefix repeating exclusively.
    Example: 123123 and abcabcabc are prefix repeating 1231234 and ababde are not.
    :return: 
    """
    i = (__domain + __domain).find(__domain, 1, -1)
    return [0] if i == -1 else [1]


def _char_diversity():
    """计算字符多样性，来一个字符统计一个，没来的默认0
    
    counts different characters, divided by domain length. 
    :return: 
    """
    counter = defaultdict(int)

    domain = __joined_dot_split_suffix_free
    for c in domain:
        counter[c] += 1

    return [len(counter)/len(domain)]


def _contains_tld_as_infix():
    """寻找是否在前缀中出现TLD内容
    
    Checks for infixes that are valid TLD endings like .de in 123.de.rwth-aachen.de
    If such a infix is found 1 is returned, 0 else
    :return:
    """
    for tld in VALID_TLDS:
        if tld[1:] in __dot_split_suffix_free:
            return [1]
    return [0]


def _n_grams():
    """分别计算123-gram结果
    
    Calculates various statistical features over the 1-,2- and 3-grams of the suffix and dot free domain
    :return: 
    """
    global __unigram
    feature = []

    for i in range(1,4):
        ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(i, i))
        counts = ngram_vectorizer.build_analyzer()(__joined_dot_split_suffix_free)
        # print(counts)
        # print(list(Counter(counts).values()))
        npa = numpy.array(list(Counter(counts).values()), dtype=int)
        # print("npa:", npa)
        if i == 1:
            __unigram = npa

        feature += __stats_over_n_grams(npa)
        # print(feature)

    return [feature]  # 修改


def __stats_over_n_grams(npa):
    """计算n-gram的特征，7个一组
    
    Calculates statistical features over ngrams decoded in numpy arrays
    stddev, median, mean, min, max, quartils, alphabetsize (length of the ngram)
    :param npa: 
    :return: 
    """
    # TODO check for Hoover Index, Gini Coef, Rosenbluth-Index, Wölbung and Schiefe

    if npa.size > 0:
        stats = [npa.std(), numpy.median(npa), npa.mean(), numpy.min(npa), numpy.max(npa), numpy.percentile(npa, 25),
             numpy.percentile(npa, 75)]
    else:
        stats = [-1, -1, -1, -1, -1, -1, -1]

    return [stats]


def _alphabet_size():
    """用到的字母个数
    
    Calculates the alphabet size of the domain
    :return: 
    """
    if len(__unigram) == 0:
        raise UnsuitableFeatureOrderException('The feature _n_grams has to be calculated before.')
    return [len(__unigram)]


def _shannon_entropy():
    """香农熵
    
    Calculates the Shannon Entropy based on the frequencies of characters.
    :return: 
    """
    # Note for me: checked against an online calculator for verification: http://www.shannonentropy.netmark.pl/
    if len(__unigram) == 0:
        raise UnsuitableFeatureOrderException('The feature _n_grams has to be calculated before.')

    return [stats.entropy(__unigram, base=2)]


def _hex_part_ratio():
    """计算全十六进制的层级占总层级的比例
    
    Counts all parts that are only hex. Normalized by the overall part count
    :return: 
    """
    hex_parts = 0
    for p in __dot_split_suffix_free:
        if all(c in HEX_DIGITS for c in p):
            hex_parts += 1

    return [hex_parts / len(__dot_split_suffix_free)]


def _underscore_ratio():
    """下划线占总字符数的比例
    
    Calculates the ratio of occuring underscores in all domain parts excluding the public suffix
    :return: 
    """
    underscore_counter = 0
    for c in __joined_dot_split_suffix_free:
        if c == '_':
            underscore_counter += 1

    return [underscore_counter / len(__joined_dot_split_suffix_free)]


def _ratio_of_repeated_chars():
    """重复字符占所有字符的比例
    
    Calculates the ratio of characters repeating in the string
    :return: 
    """
    # TODO maybe weighted? check the impact
    if len(__unigram) == 0:
        raise UnsuitableFeatureOrderException('The feature _n_grams has to be calculated before.')

    repeating = 0
    for i in __unigram:
        if i > 1:
            repeating += 1
    return [repeating / len(__unigram)]


def _consecutive_consonant_ratio():
    """连续辅音占总长度比例
    
    Calculates the ratio of conescutive consonants
    :return: 
    """
    # TODO weighted: long sequences -> higher weight

    consecutive_counter = 0
    for p in __dot_split_suffix_free:
        counter = 0
        i = 0
        for c in p:
            if c.isalpha() and c not in VOWELS:
                counter +=1
            else:
                if counter > 1:
                    consecutive_counter += counter
                counter = 0
            i += 1
            if i == len(p) and counter > 1:
                consecutive_counter += counter

    return [consecutive_counter / len(__joined_dot_split_suffix_free)]


def _consecutive_digits_ratio():
    """连续数字占总长度的比例
    
    Calculates the ratio of consecutive digits
    :return: 
    """

    consecutive_counter = 0
    for p in __dot_split_suffix_free:
        counter = 0
        i = 0
        for c in p:
            if c.isdigit():
                counter +=1
            else:
                if counter > 1:
                    consecutive_counter += counter
                counter = 0
            i += 1
            if i == len(p) and counter > 1:
                consecutive_counter += counter

    return [consecutive_counter / len(__joined_dot_split_suffix_free)]


# TODO: 以下特征将有助于解决参数子域名问题
def _contains_ip_addr():
    """是否可能包含IPv4或IPv6地址，若长度与字符范围匹配则也视为包含ip
    
    check if the domain contains a valid IP address. Considers both, v4 and v6
    :return:
    """
    match_v4 = re.search(ipv4_pattern, __domain)
    match_v6 = re.search(ipv6_pattern, __domain)
    if match_v4 or match_v6:
        return [1]
    else:
        return [0]

def _contains_uuid():
    if re.search(uuid_pattern, __domain) or re.search(reuuid_pattern, __domain):
        return [1]
    else:
        return [0]

def _contains_hex_32():
    """是否包含连续的32位十六进制数（含.），可能相关的参数是uuid、ipv6地址

    Returns:
        list: 1——包含，0——不包含
    """    
    if re.search(hex_32_pattern, __joined_dot_split_suffix_free):
        return [1]
    else:
        return [0]
    
def _contains_keywords():
    """是否包含已收集关键词，TODO:关键词列表的更新

    Returns:
        list : 包含参数子域名关键词+1，包含普通子域名关键词-1
    """    
    # for each_para_keyword in PARAMETER_KEYWORDS:
    #     for each_dot_split in __dot_split_suffix_free:
    #         if each_para_keyword in each_dot_split:
    #             return [2]
    # for each_normal_keyword in NORMAL_KEYWORDS:
    #     for each_dot_split in __dot_split_suffix_free:
    #         if each_normal_keyword in each_dot_split:
    #             return [0]
    suspect_value = 0
    for each_para_keyword in PARAMETER_KEYWORDS:
        suspect_value += __domain.count(each_para_keyword)
        
    for each_normal_keyword in NORMAL_KEYWORDS:
        suspect_value -= __domain.count(each_normal_keyword)
    return [suspect_value]

def _contains_base64():
    """判断是否有base64编码的内容，但误判高

    Args:
        s (string): 判断的字符串

    Returns:
        int: 含base64则返回1，不含则返回0
    """
    base64Pattern = "^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{4}|[A-Za-z0-9+/]{3}\-|[A-Za-z0-9+/]{2}\-\-)$"
    for each_level in __dot_split_suffix_free:
        match_base64 = re.search(base64Pattern, each_level)
        if match_base64:
            return [1]
    return [0]

def _contains_timestamp():
    for p in __dot_split_suffix_free:
        only_digits = True
        for c in p:
            if c.isalpha():
                only_digits = False
                break
        if only_digits:
            if int(p) > 1000000000 and int(p) < NOW_TIMESTAMP + 1000000000:
                return [1]
    return [0]

def _length_to_max_ratio():
    return [len(__domain)/253]

def _dot_to_max_ratio():
    return [__domain.count('.') / 253]

def _dot_amount():
    return [__domain.count('.')]

def _dot_ratio():
    return [__domain.count('.') / len(__domain)]

def _1_level_domain():
    return ['.'.join(__dot_split[-1:])]

def _2_level_domain():
    return ['.'.join(__dot_split[-2:])]

def _3_level_domain():
    return ['.'.join(__dot_split[-3:])]

def _4_level_domain():
    return ['.'.join(__dot_split[-4:])]

def _5_level_domain():
    return ['.'.join(__dot_split[-5:])]

ALL_FEATURES = _length, _vowel_ratio, _digit_ratio, _contains_digits, \
                        _has_valid_tld, _contains_one_char_subdomains, _contains_wwwdot, _subdomain_lengths_mean, \
                        _prefix_repetition, _char_diversity, _contains_subdomain_of_only_digits, _contains_tld_as_infix, \
                        _parts, _n_grams, \
                        _hex_part_ratio, _underscore_ratio, _alphabet_size, _shannon_entropy, \
                        _ratio_of_repeated_chars, _consecutive_consonant_ratio, _consecutive_digits_ratio, \
                        _contains_ip_addr  #, _contains_uuid, _contains_hex_32, _contains_keywords, _length_to_max_ratio, _dot_to_max_ratio, _dot_ratio, _dot_amount # , _1_level_domain, _2_level_domain, _3_level_domain, _4_level_domain, _5_level_domain, 
                        # 此行是修改过或新增的(修改了_contains_ip_addr)
                        # _contains_base64, _contains_timestamp未使用

def extract_all_features(data, n_jobs=-1):
    """对多个域名提取所有可用特征
    
    Function extracting all available features to a numpy feature array.
    :param data: iterable containing domain name strings
    :return: feature matrix as numpy array
    """
    parallel = Parallel(n_jobs=n_jobs, verbose=1)
    data = data.apply(lambda x: str(x))
    feature_matrix = parallel(
        delayed(extract_features)(d, ALL_FEATURES)
        for d in data
        )
    feature_df = pd.DataFrame(feature_matrix)
    columns = ["fqdn"]
    for each_feature in ALL_FEATURES:
        columns.append(each_feature.__name__)
    feature_df.columns = columns
    return pd.DataFrame(feature_df)


def extract_all_features_single(d):
    """对单个域名提取所有可用特征
    
    Extracts all features of a single domain name
    :param d: string, domain name
    :return: extracted features as numpy array
    """
    return numpy.array(extract_features(d, ALL_FEATURES))


def split_col(data, column):
    """拆分成列

    :param data: 原始数据
    :param column: 拆分的列名
    :type data: pandas.core.frame.DataFrame
    :type column: str
    """
    data = deepcopy(data)
    max_len = max(list(map(len, data[column].values)))  # 最大长度
    new_col = data[column].apply(lambda x: x + [None]*(max_len - len(x)))  # 补空值，None可换成np.nan
    new_col = numpy.array(new_col.tolist()).T  # 转置
    if new_col.ndim == 2 :
        for i, j in enumerate(new_col):
            data[column + "_" + str(i)] = j
    if new_col.ndim == 3:
        for i, j in enumerate(new_col):
            idx = 0
            for k in j:
                data[column + "_" + str(i) + str(idx)] = k
                idx += 1
    data = data.drop(columns=column)
    return data

def split_all_col(data, column_list):
    """拆分多列

    :param data: 原始数据
    :param column_list: 拆分的列名列表
    :type data: pandas.core.frame.DataFrame
    :type column_list: list
    """
    data = deepcopy(data)
    for each_column in column_list:
        print("Split features in list [ ", each_column, " ]")
        data = split_col(data, each_column)
    return data

def get_fqdn_feature_from_strange(strange_path):
    """从未知文件中提取特征"""
    strange_file = pd.read_csv(strange_path, index_col=0)
    fqdn = list(strange_file["fqdn"])
    fqdn_feature = extract_all_features(fqdn)
    for each_feature in LIST_FEATURES:
        print("Split features in list [ ", each_feature, " ]")
        fqdn_feature = split_col(fqdn_feature, each_feature)
    return fqdn_feature


def get_fqdn_feature_from_strange_and_append(strange_path):
    strange_file = pd.read_csv(strange_path, index_col=0)
    fqdn = list(strange_file["fqdn"])
    fqdn_feature = extract_all_features(fqdn)
    for each_feature in LIST_FEATURES:
        print("Split features in list [ ", each_feature, " ]")
        fqdn_feature = split_col(fqdn_feature, each_feature)
    fqdn_feature = pd.concat([strange_file, fqdn_feature],axis=1)
    fqdn_feature = fqdn_feature.loc[:,~fqdn_feature.columns.duplicated()]
    return fqdn_feature