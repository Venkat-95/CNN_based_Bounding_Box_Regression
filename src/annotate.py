import os
import cv2
import json
from pathlib import Path

from xml.etree import ElementTree as ET

CURRENT_FILE_PATH = Path(__file__).resolve()
DATA_FILE_PATH = CURRENT_FILE_PATH.parent.parent / 'data'


def check_directory(file_path: Path) -> tuple[bool, bool]:
    is_directory = False
    files_present = False

    try:
        if os.listdir(file_path):
            is_directory = True
            files_present = True
    except:
        files_present = False
    return is_directory, files_present


class Annotate:
    def __init__(self):
        self.path_images: Path = DATA_FILE_PATH / 'images'
        self.path_annotations: Path = DATA_FILE_PATH / 'annotations_XML'
        self.is_images_directory, self.images_present = check_directory(self.path_images)
        self.is_annotations_directory, self.annotations_present = check_directory(self.path_annotations)
        if not self.is_annotations_directory or not self.annotations_present:
            self.path_annotations = DATA_FILE_PATH / 'annotations_json'
            self.is_annotations_directory, self.annotations_present = check_directory(self.path_annotations)

    def draw_annotation(self, event, x, y, flags, param):
        global ix, iy, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(self.img, (ix, iy), (x, y), (0, 255, 0), 2)
            print(ix, iy, x, y)
            self.img_data['bboxes'].append((ix, iy, x, y))

    def process_images(self, path_images: Path):
        data_annotations = []
        image_files = list(self.path_images.glob('*.png'))
        os.makedirs(DATA_FILE_PATH / 'annotations_json',exist_ok=True)
        self.path_annotations = DATA_FILE_PATH / 'annotations_json'
        for img_file in image_files:
            print('file:', img_file)
            img_data = {}
            img_data['filepath'] = os.path.basename(img_file)
            img_data['class'] = 'license'
            img_data['bboxes'] = []
            self.img = cv2.imread(str(img_file))
            height, width, channels = self.img.shape
            img_data['size'] = (height, width, channels)
            self.img_data = img_data
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('image', self.draw_annotation)
            check = 0
            while (1):
                cv2.imshow('image', self.img)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    check = 1
                    break
                elif k == ord('q'):
                    break
            # End of draw_annotation
            cv2.destroyAllWindows()
            if (len(img_data['bboxes']) > 0):
                data_annotations.append(img_data)
            if (check):
                break
        with open(os.path.join(self.path_annotations,'annotation_data.json'), 'w') as f:
            json.dump(data_annotations, f, indent=2)

    '''
    Following are static methods because the functionality is called from
    the child class preprocessing.py-> ImageDataGenerator
    '''

    @staticmethod
    def get_json_data(path_annotations: Path):
        data_dict = {}
        img_info_dict = {}
        with open(path_annotations / "annotation_data.json", 'r') as json_file:
            data_json = json.load(json_file)
        for i in data_json:
            _, img_name = os.path.split(i['filepath'])
            data_dict[img_name] = [i['class'], i['bboxes']]
            img_info_dict[img_name] = i['size']
        return data_dict, img_info_dict

    @staticmethod
    def get_xml_data(path_annotations: Path):
        data_dict = {}
        img_info_dict = {}
        XML_file_list = path_annotations.glob("*.xml")
        for file_path in XML_file_list:
            _, file_name = os.path.split(file_path)
            data_xml = ET.parse(file_path)
            myroot = data_xml.getroot()
            img_name = str(myroot.find("filename").text)
            height = int(myroot.find("size")[0].text)
            width = int(myroot.find("size")[1].text)
            channels = int(myroot.find("size")[2].text)
            bbox_coordinates = []
            for member in myroot.findall('object'):
                class_name = member[0].text  # class name
                # bbox coordinates
                xmin = int(member[5][0].text)
                ymin = int(member[5][1].text)
                xmax = int(member[5][2].text)
                ymax = int(member[5][3].text)
                # store data in list
                bbox_coordinates.append([class_name, [xmin, ymin, xmax, ymax]])
                img_info_dict[img_name] = [height, width, channels]
                data_dict[img_name] = bbox_coordinates
        # pprint(data_dict)
        return data_dict, img_info_dict
