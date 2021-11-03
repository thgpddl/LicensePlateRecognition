import numpy as np
import math
import cv2


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"

        self.character_str = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
        if use_space_char:
            self.character_str.append(" ")
        dict_character = list(self.character_str)

        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                        batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                                                        idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class PostProcess(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path="LicensePlateRecognizer/assets/plate_dict.txt",
                 use_space_char=False):
        super(PostProcess, self).__init__(character_dict_path, use_space_char=use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class PreProcess:
    def __init__(self):
        print("")

    def __call__(self, img):
        imgC, imgH, imgW = (3, 32, 320)  # 规定尺寸[3,32,320]
        assert imgC == img.shape[2]  # 通道是否为3
        h, w = img.shape[:2]  # 获取高宽
        ratio = w / float(h)  # 获取宽高比
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))  # 以高不变，保持比例
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255  # →[c,h,w]，[0~1]
        resized_image -= 0.5
        resized_image /= 0.5  # [-1~1]
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        padding_im = padding_im[np.newaxis, :]  # 升维[1,3,32,320]
        return padding_im
