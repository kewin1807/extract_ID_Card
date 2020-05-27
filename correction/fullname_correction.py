import os
from collections import defaultdict
from correction.utils import StringDistance, extract_digit


class FullnameCorrection:
    '''
    Fullname correction with phrase compare
    '''

    def __init__(self, cost_dict_path=None, fullnames_path=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if cost_dict_path is None:
            cost_dict_path = os.path.join(
                dir_path, 'data', 'cost_char_dict.txt')
        if fullnames_path is None:
            fullnames_path = os.path.join(dir_path, 'data', 'fullnames.txt')
        self.string_distance = StringDistance(cost_dict_path=cost_dict_path)
        self.fullnames = []
        with open(fullnames_path, 'r', encoding='utf-8') as f:
            for line in f:
                entity = line.strip()
                if not entity:
                    break
                entity = entity.split('|')
                self.fullnames.extend(entity)

    def correct(self, phrase, correct_phrases, nb_candidates=3, distance_threshold=40):
        candidates = [(None, distance_threshold)] * nb_candidates
        max_diff_length = distance_threshold
        for correct_phrase in correct_phrases:
            if abs(len(phrase) - len(correct_phrase)) >= max_diff_length:
                continue
            else:
                distance = self.string_distance.distance(
                    phrase, correct_phrase)
            if distance < candidates[-1][1]:
                candidates[-1] = (correct_phrase, distance)
                candidates.sort(key=lambda x: x[1])
        return candidates

    def correction(self, fullname, correct_th=50):
        '''
        Fullname should be in format: Ho dem ten
        and only contain characters
        Return: (corrected_fullname: str, distance: integer)
            corrected_fullname: fullname after corrected. In case fullname can't corrected, return
            input fullname
            distance: distance between corrected fullname and input fullname. In case fullname
            can't correct, return -1
        '''
        if not isinstance(fullname, str):
            raise ValueError('Fullname must be a string')

        fullname_candidates = self.correct(fullname, self.fullnames)
        print(fullname_candidates)
        fullname_candidates.sort(key=lambda x: x[1])
        # get the first element with smallest distance
        if len(fullname_candidates):
            result, distance_result = fullname_candidates[0]
            if distance_result <= correct_th:
                return result, distance_result

        return fullname, -1
